import cardinality.DynamicDemiLog;
import ddl.DDLComparison;
import ddl.DDLIndexBase;
import ddl.DDLLoader;
import ddl.DDLLoaderMT;
import ddl.DDLRecord;
import fileIO.FileFormat;
import fileIO.ReadWrite;
import shared.Shared;
import stream.Read;
import stream.Streamer;
import stream.StreamerFactory;
import structures.ListNum;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntConsumer;

/**
 * BBTools DynamicDemiLog lanes for the micro comparison, driven through BBTools' own classes.
 *
 * <pre>
 * sketch LIST OUT THREADS             DDLWriter's per-file sketching: one k=25 sketch with 2048
 *                                     buckets and exponent 5 per listed file
 * compare REFS QUERIES|- OUT THREADS  one little-endian f32 ANI per pair: row-major query x
 *                                     reference, or the strict upper triangle of REFS for '-'
 * search REFS QUERIES|- K OUT THREADS DDLCompare's indexed search: index match counts prefilter
 *                                     references, full comparisons rank them, and the K best
 *                                     are written as 'query reference wkid' index rows
 * </pre>
 *
 * REFS and QUERIES are sketch files written by {@code sketch}; the list file holds one FASTX path
 * per line, which avoids BBTools' single comma-joined {@code in=} argument. With
 * CUDDL_RESIDENT_TIMINGS set, every command also writes its in-memory time: hashing staged
 * reads, comparing loaded sketches, or searching them, with the index build time apart.
 */
public class BBToolsMicro {

	static final int K=25;
	/** DDLCompare's default minimum index matches before a full comparison. */
	static final int MIN_HITS=5;
	/** DDLWriter's default hash seed. */
	static final long SEED=12345L;
	/** FASTX bytes on disk staged per sketching batch. */
	static final long RESIDENT_BYTES=1L<<30;

	public static void main(String[] args) throws Exception {
		switch(args[0]){
			case "sketch" -> sketch(Path.of(args[1]), args[2], Integer.parseInt(args[3]));
			case "compare" -> compare(args[1], args[2], Path.of(args[3]), Integer.parseInt(args[4]));
			case "search" -> search(args[1], args[2], Integer.parseInt(args[3]), Path.of(args[4]),
				Integer.parseInt(args[5]));
			default -> throw new IllegalArgumentException("unknown command "+args[0]);
		}
	}

	/**
	 * DDLWriter's per-file sketching with the reading split out: files are read into memory in
	 * batches of at most RESIDENT_BYTES on disk, and only hashing the staged reads into each
	 * file's DynamicDemiLog counts as resident time.
	 */
	static void sketch(Path list, String out, int threads) throws Exception {
		List<String> files=Files.readAllLines(list).stream().filter(l -> !l.isEmpty()).toList();
		ReadWrite.USE_PIGZ=ReadWrite.USE_UNPIGZ=true;
		Shared.capBufferLen(200);
		Shared.capBuffers(8);
		Shared.setThreads(threads);
		DynamicDemiLog.setExponent(5);
		ArrayList<DDLRecord> records=new ArrayList<>();
		long resident=0;
		for(int start=0; start<files.size();){
			int end=start;
			long staged=0;
			while(end<files.size() && (end==start || staged<RESIDENT_BYTES)){
				staged+=Files.size(Path.of(files.get(end++)));
			}
			final int first=start;
			ArrayList<ArrayList<Read>> reads=new ArrayList<>();
			for(int f=first; f<end; f++){reads.add(null);}
			parallel(end-first, threads, i -> reads.set(i, readAll(files.get(first+i))));
			DynamicDemiLog[] sketches=new DynamicDemiLog[end-first];
			long tick=System.nanoTime();
			parallel(end-first, threads, i -> {
				DynamicDemiLog ddl=DynamicDemiLog.create(2048, K, SEED, 0f, false, false);
				for(Read r : reads.get(i)){ddl.hash(r);}
				sketches[i]=ddl;
			});
			resident+=System.nanoTime()-tick;
			for(int i=0; i<sketches.length; i++){
				String path=files.get(first+i);
				String name=new File(path).getName();
				DDLRecord rec=new DDLRecord(sketches[i], first+i, -1, name);
				rec.filename=name;
				for(Read r : reads.get(i)){
					rec.bases+=r.pairLength();
					rec.contigs++;
				}
				rec.cardinality=sketches[i].cardinality();
				records.add(rec);
			}
			start=end;
		}
		DDLLoader.writeFile(records, out, true, K, SEED);
		emitResident("sequence_ascii", resident, -1);
	}

	static ArrayList<Read> readAll(String path){
		FileFormat ff=FileFormat.testInput(path, FileFormat.FASTA, null, true, true);
		Streamer cris=StreamerFactory.getReadInputStream(-1, false, ff, null, -1);
		cris.start();
		ArrayList<Read> reads=new ArrayList<>();
		for(ListNum<Read> ln=cris.nextList(); ln!=null && ln.list!=null && !ln.list.isEmpty();
				ln=cris.nextList()){
			reads.addAll(ln.list);
			cris.returnList(ln);
		}
		ReadWrite.closeStreams(cris);
		return reads;
	}

	/** DDLLoaderMT loads records in completion order; the sketch ids are the list positions. */
	static ArrayList<DDLRecord> load(String path, int threads){
		ArrayList<DDLRecord> records=DDLLoaderMT.loadFile(path, K, threads);
		records.sort(Comparator.comparingLong(r -> r.id));
		return records;
	}

	/** Runs body(i) for every i in [0, n) on @p threads workers, rows handed out one at a time. */
	static void parallel(int n, int threads, IntConsumer body) throws Exception {
		AtomicInteger next=new AtomicInteger();
		ExecutorService pool=Executors.newFixedThreadPool(threads);
		ArrayList<Future<?>> futures=new ArrayList<>();
		for(int t=0; t<threads; t++){
			futures.add(pool.submit(() -> {
				for(int i=next.getAndIncrement(); i<n; i=next.getAndIncrement()){body.accept(i);}
			}));
		}
		for(Future<?> f : futures){f.get();}
		pool.shutdown();
	}

	static void compare(String refPath, String queryPath, Path out, int threads) throws Exception {
		ArrayList<DDLRecord> refs=load(refPath, threads);
		boolean all=queryPath.equals("-");
		ArrayList<DDLRecord> queries=all ? refs : load(queryPath, threads);
		int n=refs.size();
		long[] offsets=new long[queries.size()+1];
		for(int q=0; q<queries.size(); q++){
			offsets[q+1]=offsets[q]+(all ? n-1-q : n);
		}
		float[] ani=new float[Math.toIntExact(offsets[queries.size()])];
		long tick=System.nanoTime();
		parallel(queries.size(), threads, q -> {
			DDLComparison working=new DDLComparison();
			int at=(int)offsets[q];
			for(int r=all ? q+1 : 0; r<n; r++){
				ani[at++]=working.compare(queries.get(q), refs.get(r), K).ani;
			}
		});
		long resident=System.nanoTime()-tick;
		ByteBuffer bytes=ByteBuffer.allocate(4*ani.length).order(ByteOrder.LITTLE_ENDIAN);
		bytes.asFloatBuffer().put(ani);
		Files.write(out, bytes.array());
		emitResident("sketches", resident, -1);
	}

	static void search(String refPath, String queryPath, int k, Path out, int threads)
			throws Exception {
		ArrayList<DDLRecord> refs=load(refPath, threads);
		boolean all=queryPath.equals("-");
		ArrayList<DDLRecord> queries=all ? refs : load(queryPath, threads);
		int n=refs.size();
		IdentityHashMap<DDLRecord, Integer> position=new IdentityHashMap<>();
		for(int r=0; r<n; r++){position.put(refs.get(r), r);}
		long tick=System.nanoTime();
		DDLIndexBase index=DDLIndexBase.create(refs.get(0).ddl.buckets);
		index.addAll(refs, threads);
		long build=System.nanoTime()-tick;
		int[][] hits=new int[queries.size()][];
		float[][] scores=new float[queries.size()][];
		tick=System.nanoTime();
		parallel(queries.size(), threads, q -> {
			DDLRecord query=queries.get(q);
			int[] counts=index.query(query.ddl);
			ArrayList<DDLComparison> ranked=new ArrayList<>();
			// DDLCompare's CompareThread: references reaching MIN_HITS are compared in full; when
			// that leaves fewer than its buffer, references with any match are added.
			int buffer=20+2*k;
			for(int pass=0; pass<2; pass++){
				for(int r=0; r<n; r++){
					if(all && r==q){continue;}
					boolean take=pass==0 ? counts[r]>=MIN_HITS : counts[r]>=1 && counts[r]<MIN_HITS;
					if(take){ranked.add(new DDLComparison().compare(query, refs.get(r), K));}
				}
				if(ranked.size()>=buffer){break;}
			}
			ranked.sort(null);
			int kept=Math.min(k, ranked.size());
			hits[q]=new int[kept];
			scores[q]=new float[kept];
			for(int i=0; i<kept; i++){
				hits[q][i]=position.get(ranked.get(i).refRecord);
				scores[q][i]=ranked.get(i).wkid;
			}
		});
		long resident=System.nanoTime()-tick;
		try(BufferedWriter w=Files.newBufferedWriter(out)){
			for(int q=0; q<queries.size(); q++){
				for(int i=0; i<hits[q].length; i++){
					w.write(q+"\t"+hits[q][i]+"\t"+scores[q][i]+"\n");
				}
			}
		}
		emitResident("indexed_sketches", resident, build);
	}

	static void emitResident(String input, long nanos, long indexBuildNanos) throws IOException {
		String path=System.getenv("CUDDL_RESIDENT_TIMINGS");
		if(path==null){return;}
		String extra=indexBuildNanos<0 ? "" : ",\"index_build_ms\":"+indexBuildNanos/1e6;
		Files.writeString(Path.of(path), "{\"resident_ms\":"+nanos/1e6
			+",\"source\":\"steady_clock_cpu_wall\",\"device\":\"cpu\",\"input\":\""+input+"\""
			+extra+"}");
	}
}
