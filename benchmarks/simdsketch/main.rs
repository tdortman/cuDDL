//! SimdSketch micro-benchmark lanes, driven through the `simd-sketch` library.
//!
//! `sketch` turns FASTX genomes into one sketch file. `compare` loads that file and scores
//! FASTX queries against it, or every unordered reference pair when no queries are given, and
//! writes one little-endian `f32` Jaccard similarity per pair: row-major query x reference, or
//! the strict upper triangle row by row.
//!
//! With `CUDDL_RESIDENT_TIMINGS` set, each command also writes the time spent on in-memory work
//! alone: packing and sketching staged sequences, or comparing loaded sketches.

use clap::{Parser, Subcommand};
use packed_seq::PackedNSeqVec;
use rayon::prelude::*;
use simd_sketch::{Sketch, SketchAlg, SketchParams, Sketcher};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

type Result<T, E = Box<dyn Error + Send + Sync>> = std::result::Result<T, E>;

const BINCODE: bincode::config::Configuration<
    bincode::config::LittleEndian,
    bincode::config::Fixint,
> = bincode::config::standard().with_fixed_int_encoding();

#[derive(Parser)]
struct Args {
    /// Worker threads; defaults to all available processors.
    #[arg(short = 'j', long, global = true)]
    threads: Option<usize>,
    /// FASTX bytes on disk staged per resident batch. Sketches stay corpus-sized.
    #[arg(long, global = true, default_value_t = 1 << 30)]
    resident_bytes: u64,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Sketch the FASTX files listed in INPUTS into one sketch file.
    Sketch {
        #[arg(short, default_value_t = 25)]
        k: usize,
        #[arg(short, default_value_t = 2048)]
        s: usize,
        /// Low bits kept per bucket.
        #[arg(short, default_value_t = 8)]
        b: usize,
        #[arg(long)]
        output: PathBuf,
        /// File with one FASTX path per line.
        inputs: PathBuf,
    },
    /// Compare FASTX queries, or every reference pair, against a sketch file.
    Compare {
        #[arg(long)]
        references: PathBuf,
        /// File with one query FASTX path per line; omit for all reference pairs.
        #[arg(long)]
        queries: Option<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
}

/// Bucket sketch of canonical k-mers. K-mers with a non-ACGT base are skipped, as every other
/// lane does, and empty buckets are excluded from the similarity.
fn sketcher(k: usize, s: usize, b: usize) -> Sketcher {
    SketchParams {
        alg: SketchAlg::Bucket,
        rc: true,
        k,
        s,
        b,
        filter_empty: true,
        filter_out_n: true,
    }
    .build()
}

fn read_list(path: &Path) -> Result<Vec<PathBuf>> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(text
        .lines()
        .filter(|l| !l.is_empty())
        .map(PathBuf::from)
        .collect())
}

fn parse(path: &Path) -> Result<Vec<Vec<u8>>> {
    let mut reader =
        needletail::parse_fastx_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut records = Vec::new();
    while let Some(record) = reader.next() {
        let record = record.map_err(|e| format!("{}: {e}", path.display()))?;
        records.push(record.seq().into_owned());
    }
    Ok(records)
}

/// Sketches @p paths in batches of at most @p cap bytes on disk, so staged sequences stay
/// bounded on a large corpus. Reading and parsing a batch stay outside @p resident; packing
/// and sketching it are inside.
fn sketch_files(
    sketcher: &Sketcher,
    paths: &[PathBuf],
    cap: u64,
    resident: &mut Duration,
) -> Result<Vec<Sketch>> {
    let mut sketches = Vec::with_capacity(paths.len());
    let mut start = 0;
    while start < paths.len() {
        let mut end = start;
        let mut staged = 0;
        while end < paths.len() && (end == start || staged < cap) {
            staged += fs::metadata(&paths[end])
                .map_err(|e| format!("{}: {e}", paths[end].display()))?
                .len();
            end += 1;
        }
        let batch = paths[start..end]
            .par_iter()
            .map(|path| parse(path))
            .collect::<Result<Vec<_>>>()?;
        let tick = Instant::now();
        sketches.par_extend(batch.par_iter().map(|records| {
            let packed: Vec<_> = records
                .iter()
                .map(|r| PackedNSeqVec::from_ascii(r))
                .collect();
            let slices: Vec<_> = packed.iter().map(|p| p.as_slice()).collect();
            sketcher.sketch_seqs(&slices)
        }));
        *resident += tick.elapsed();
        start = end;
    }
    Ok(sketches)
}

fn emit_resident(input: &str, elapsed: Duration) -> Result<()> {
    let Some(path) = std::env::var_os("CUDDL_RESIDENT_TIMINGS") else {
        return Ok(());
    };
    fs::write(
        path,
        format!(
            "{{\"resident_ms\":{},\"source\":\"steady_clock_cpu_wall\",\"device\":\"cpu\",\
             \"input\":\"{input}\"}}",
            elapsed.as_secs_f64() * 1e3
        ),
    )?;
    Ok(())
}

fn sketch(k: usize, s: usize, b: usize, cap: u64, inputs: &Path, output: &Path) -> Result<()> {
    let paths = read_list(inputs)?;
    let mut resident = Duration::ZERO;
    let sketches = sketch_files(&sketcher(k, s, b), &paths, cap, &mut resident)?;
    let mut file = BufWriter::new(File::create(output)?);
    bincode::encode_into_std_write(&sketches, &mut file, BINCODE)?;
    file.flush()?;
    emit_resident("sequence_ascii", resident)
}

fn compare(cap: u64, references: &Path, queries: Option<&Path>, output: &Path) -> Result<()> {
    let refs: Vec<Sketch> =
        bincode::decode_from_std_read(&mut BufReader::new(File::open(references)?), BINCODE)?;
    let n = refs.len();
    let mut similarities;
    let tick;
    if let Some(queries) = queries {
        let params = refs
            .first()
            .ok_or("empty reference sketch file")?
            .to_params();
        let sketcher = sketcher(params.k, params.s, params.b);
        // Query sketching is wall time only; the resident interval is the comparison.
        let mut sketching = Duration::ZERO;
        let queries = sketch_files(&sketcher, &read_list(queries)?, cap, &mut sketching)?;
        similarities = vec![0f32; queries.len() * n];
        tick = Instant::now();
        similarities
            .par_chunks_mut(n.max(1))
            .zip(&queries)
            .for_each(|(row, query)| {
                for (out, reference) in row.iter_mut().zip(&refs) {
                    *out = query.jaccard_similarity(reference);
                }
            });
    } else {
        similarities = vec![0f32; n * n.saturating_sub(1) / 2];
        let mut rows = Vec::with_capacity(n);
        let mut rest = similarities.as_mut_slice();
        for i in 0..n {
            let (row, tail) = rest.split_at_mut(n - 1 - i);
            rows.push(row);
            rest = tail;
        }
        tick = Instant::now();
        rows.into_par_iter().enumerate().for_each(|(i, row)| {
            for (out, reference) in row.iter_mut().zip(&refs[i + 1..]) {
                *out = refs[i].jaccard_similarity(reference);
            }
        });
    }
    let resident = tick.elapsed();
    let mut file = BufWriter::new(File::create(output)?);
    for value in &similarities {
        file.write_all(&value.to_le_bytes())?;
    }
    file.flush()?;
    emit_resident("sketches", resident)
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()?;
    }
    match args.command {
        Command::Sketch {
            k,
            s,
            b,
            output,
            inputs,
        } => sketch(k, s, b, args.resident_bytes, &inputs, &output),
        Command::Compare {
            references,
            queries,
            output,
        } => compare(
            args.resident_bytes,
            &references,
            queries.as_deref(),
            &output,
        ),
    }
}
