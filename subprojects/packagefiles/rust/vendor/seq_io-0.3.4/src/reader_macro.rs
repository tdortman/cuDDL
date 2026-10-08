

macro_rules! impl_reader_methods {
    ($rdr:ty, $rset:ty, $record:ty, $error:ty) => {
        impl $rdr {
            /// Searches the next FASTQ record and returns a
            ///#[doc = $refrec_link]
            /// exactly like [`next()`](Reader::next), but this function does not
            /// compare the lengths of the sequence and quality scores. The check can
            /// (and should!) be done later using [`check_lengths()`](
            ///#[doc = $record_link]
            /// ).
            ///
            /// **Note**: not checking lengths this with multi-line FASTQ is a very bad
            /// idea, since the parser uses the sequence length as orientation when
            /// parsing the quality scores; quality lengths will never be smaller,
            /// but they can be larger than sequence lengths, and not doing a length
            /// check may lead to confusing errors later, or in the worst case
            /// "swallowing" of a record.
            // TODO: handle...
            #[inline]
            fn next_unchecked(&mut self) -> Option<error::Result<Self::Record<'_>>> {
                // if let Some(res) = self.next_inner() {
                //     res.map(|_| {
                //         let rec: RefRecord<F, P> = rdata.ref_record(buf, self.buf_reader.file_offset());
                //         rec

                //     })
                // }
                self.read_next().map(|res| {
                    res.map(move |_| {
                        self.record_data
                            .get_ref_record(self.buf_reader.buffer(), self.buf_reader.file_offset())
                    })
                })
            }

            /// Updates a
            ///#[doc = $rset_link]
            /// with new data.
            /// The contents of the internal buffer are just copied over to the record
            /// set and the positions of all records are found.
            /// Old data will be erased. Returns `None` if the input reached its end.
            /// Iterating over a record set will yield
            ///#[doc = $refrec_link]
            /// instances.
            ///
            /// # Example:
            ///
            /// ```
            /// use seq_io::prelude::*;
            ///#[doc = $import_rset]
            ///
            ///#[doc = $seq2]
            /// // Read records the normal way (for illustration, actually
            /// // we could also use the Reader::records iterator here)
            /// let mut reader = Reader::new(&seq[..]);
            /// let mut records = vec![];
            /// while let Some(res) = reader.next() {
            ///     records.push(res.unwrap().to_owned_record());
            /// }
            ///
            /// // Use read_record_set
            /// let mut reader = Reader::new(&seq[..]);
            /// let mut record_set = RecordSet::default(); // initialize once, reuse later
            /// let mut rset_records = vec![];
            /// while reader.read_record_set(&mut record_set).unwrap() {
            ///     for record in &record_set {
            ///         rset_records.push(record.to_owned_record());
            ///     }
            /// }
            ///
            /// // The two methods should give the same result
            /// assert_eq!(records, rset_records);
            /// ```
            #[inline]
            fn read_record_set(&mut self, record_set: &mut Self::RecordSet) -> error::Result<bool> {
                self.read_record_set_exact(record_set, None)
            }

            #[inline]
            fn read_record_set_exact(
                &mut self,
                record_set: &mut Self::RecordSet,
                n_records: Option<usize>,
            ) -> error::Result<bool> {
                self._read_record_set(
                    record_set, 
                    |r, d| r.set_next_pos(d), 
                    |r, b, d| {
                        r.set_data(b.buffer(), b.file_offset(), &d.opts, &d.opts_mut);
                    },
                    n_records
                )
            }

            #[inline]
            pub(crate) fn read_next(&mut self) -> Option<error::Result<()>> {
                //println!( "NEXT ========> f: {}, m: {}/{} check: {} {:?}", fasta, multiline_fasta, multiline_fastq, check_lengths, self.state);
        
                macro_rules! try_finish {
                    ($self:ident, $expr: expr) => {
                        match $expr {
                            Ok(item) => item,
                            Err(e) => {
                                $self.state = State::Finished;
                                return Some(Err(::std::convert::From::from(e)));
                            }
                        }
                    };
                }
        
                match self.state {
                    State::New => {
                        try_finish!(self, self.fill_buf());
                        self.state = State::Parsing;
                    }
                    State::Positioned => {
                        self.state = State::Parsing;
                    }
                    State::Finished => {
                        return None;
                    }
                    State::Parsing => {
                        // TODO: reset length_diff only with only with: guess_qual || multiline_fastq && !check_lengths
                        self.increment_record();
                    }
                };
        
                loop {
                    // loops until complete record is found (enlarging buffer / relocating contents if necessary)
                    match try_finish!(self, self.find_record()) {
                        SearchResult::Complete => {
                            return Some(Ok(()));
                        }
                        SearchResult::Incomplete => {
                            //println!("-> found result with {:?} check {} {:?}", search_pos, check_last_byte, self.pos);
                            // not at end -> adjust buffer and try again
                            try_finish!(self, self.adjust_buffer());
                            // -> start over searching in next iteration
                        }
                        SearchResult::End => {
                            return None;
                        }
                    }
                }
            }
        
            // add_fn is used instead of having set_next_pos in RecordSetT trait
            // to prevent the 'static lifetime bound for F and P
            // TODO: better way for this?
            #[inline]
            fn _read_record_set<Fa, Fd>(
                &mut self,
                record_set: &mut $rset,
                add_fn: Fa,
                set_data_fn: Fd,
                n_records: Option<usize>,
            ) -> error::Result<bool>
            where
                Fa: Fn(&mut S, &RecordData<F, P>),
                Fd: Fn(&mut S, &BufReader<R, B>, &RecordData<F, P>),
            {
                record_set.clear();
        
                macro_rules! try_finish {
                    ($self:ident, $expr: expr) => {
                        match $expr {
                            Ok(item) => item,
                            Err(e) => {
                                $self.state = State::Finished;
                                return Err(::std::convert::From::from(e));
                            }
                        }
                    };
                }
        
                let adjust_buffer = match self.state {
                    State::New => {
                        // Parsing not yet started, or a seek() to another position
                        // in the file was done, requiring the buffer to refilled.
                        try_finish!(self, self.fill_buf());
                        self.state = State::Positioned;
                        false
                    }
                    State::Finished => {
                        // End of in put reached, always return false
                        return Ok(false);
                    }
                    State::Parsing => {
                        // next() was previously called, the current record has
                        // already been returned -> go to next record and
                        // make sure it is moved to the start of the buffer if necessary.
                        self.increment_record();
                        self.state = State::Positioned;
                        true
                    }
                    State::Positioned => {
                        // The previous call to read_record_set() left the current record
                        // (the last in the buffer) incomplete.
                        // Or, a seek() to a position reachable from within the current
                        // buffer was done.
                        // In such a case, the current record needs to be moved to the
                        // start of the buffer, and the remaining part needs to be
                        // refilled.
                        true
                    }
                };
        
                // TODO: only make_room here
                // println!("new outer cap {} {:?} {:?} adj {}", self.buf_reader.capacity(), self.record_data, self.search_pos, adjust_buffer);
                if adjust_buffer {
                    // not at end -> adjust buffer and try again
                    try_finish!(self, self.make_room());
                    // println!("made room -> {}, off {}", self.buf_reader.capacity(), self.buf_reader.file_offset());
                }
            // println!(" => read recset at {:?} / buf {} l {} n recs {:?}", self.search_pos, self.buf_reader.file_offset(), self.buf_reader.buffer().len(), n_records);
        
                // search for records
                loop {
                    match try_finish!(self, self.find_record()) {
                        SearchResult::Complete => {
                            // complete record found
                            // record_set.set_next_pos(&self.record_data);
                            add_fn(record_set, &self.record_data);
                            //println!( "[RSET] complete: {:?}, lines {} + {}", self.pos, self.line_idx, self.pos.num_lines());
                            // initiate next record
                            self.increment_record();
                            // if exact number of records requested and number reached
                            // -> stop
                            if let Some(n) = n_records {
                                if record_set.len() == n {
                                    // // TODO: ok?
                                    // self.search_pos = Some(FieldPosition::new(
                                    //     SearchPos::HEAD,
                                    //     self.pos.record_start(),
                                    // ));
                                    break;
                                }
                            }
                        }
                        SearchResult::Incomplete => {
                            // If an incomplete record is found, we usually
                            // return the record set, unless it is still empty, or
                            // the number of requested records has not yet been matched.
                            // In such a case, we enlarge the buffer and try again.
                            let do_grow = if let Some(n) = n_records {
                                // TODO: maybe enlarge more conservatively?
                                record_set.len() < n
                            } else {
                                record_set.len() == 0
                            };
        
                            if do_grow {
                                // Enlarge the buffer and try again in next iteration
                                try_finish!(self, self.grow());
                                continue;
                            }
                            // record set completed -> return
                            break;
                        }
                        SearchResult::End => {
                            // searching finished
                            self.state = State::Finished;
                            break;
                        }
                    }
                };
        
                // println!("set buffer {:?}", self.buf_reader.buffer());
                // self.record_idx += record_set.len() as u64;
                // println!("finished rset {:?}, lines {}",record_set.len(), self.line_idx);
                set_data_fn(record_set, &self.buf_reader, &self.record_data);
                Ok(true)
            }
        
            // Invokes the call to `FormatParser::find_record` and transforms the results
            #[inline]
            fn _find_record(&mut self) -> Result<SearchResult, $error> {
                if search_buf.is_last() && search_pos.cursor == search_buf.data.len() {
                    // TODO: better way? checking lines.pos() >= search_buf.data.len()?
                    return Ok(SearchResult::End);
                }
                let mut at_end = false;  // TODO: can it work without?
                loop {
                    let mut buffer_adj = search_buf.data;
                    if (opts.multi_line_fastq() || !opts.single_line_fasta()) && !at_end {
                        // Ensure that there is always an additional byte at the end of the
                        // buffer, which is not searched (unless at end of input). This
                        // allows peeking one byte ahead if necessary.
                        buffer_adj = buffer_adj.split_last().unwrap().1;
                    }
                    // do the searching
                    let mut lines = search_pos.search(buffer_adj, b'\n');
                    // TODO: side effects of fn?
                    // - resets incomplete_pos
                    // - ?...
                    let res = self._find_record(variant, &mut lines, buf_pos, record_offset, search_pos, search_buf.data, opts);
                    // update the cursor position (also important for constructing errors)
                    search_pos.update(&lines);
                    // check the result and return or continue searching
                    // println!("found {:?} {} {:?} {:?} {:?}", res, false, buf_pos, search_pos, opts);
                    match res {
                        Ok(Complete) => {
                            return Ok(SearchResult::Complete);
                        },
                        // TODO: may be inefficient
                        Ok(Incomplete(ip)) => {
                            self.incomplete_pos = Some(ip);
                            if search_buf.is_last() {
                                // self.done = true;  // there will be max. one additional record
                                if (!opts.single_line_fasta() || opts.multi_line_fastq()) && !at_end {
                                    // In the case of multi-line formats:
                                    // The last byte was not yet checked; we need re-search
                                    // the last chunk including the last byte to check 
                                    // what is there
                                    at_end = true;
                                    search_pos.update(&lines);
                                    continue;
                                }
                                //  check if last record is present and valid
                                let has_last_record = check_end(
                                    buf_pos,
                                    search_pos,
                                    buffer_adj,
                                    ip,
                                    variant.is_fasta() == Some(true),
                                    opts,
                                )?;
                                // reset (later seek is possible)
                                self.incomplete_pos = None;
                                // println!("has last {} {:?}", has_last_record, buf_pos);
                                if has_last_record {
                                    // break,
                                    return Ok(SearchResult::Complete);
                                }
                                return Ok(SearchResult::End);
                            }
                            return Ok(SearchResult::Incomplete);
                        },
                        Err(e) => {
                            // reset the reader state in case of seek
                            self.incomplete_pos = None;
                            return Err(e);
                        }
                    }
        

                let buf = self.buf_reader.buffer();
            // println!("find buflen {} {:?} {:?}", buf.len(), self.record_data, self.search_pos);
            // println!("...buf from cursor {:?}", std::str::from_utf8(&self.buf_reader.buffer()[self.search_pos.cursor..]));
                let at_end = buf.is_empty() || (buf.len() < self.buf_reader.capacity() && self.search_pos.cursor >= buf.len());
            // println!("at end {}", at_end);
                let res = if !at_end {
                    self.find_record(
                        &mut self.record_data.pos.0,
                        &mut self.record_data.pos.1,
                        &mut self.search_pos,
                        &SearchBuffer {
                            data: buf,
                            capacity: self.buf_reader.capacity(),
                            file_offset: self.buf_reader.file_offset(),
                        },
                        &self.record_data.opts,
                        &mut self.record_data.opts_mut,
                    )
                } else {
                    Ok(SearchResult::End)
                };
                match res {
                    Ok(res) => {
                        if res == SearchResult::End {
                            self.state = State::Finished;
                        }
                        Ok(res)
                    }
                    Err(e) => {
                        self.state = State::Finished;
                        Err(e.into_parse_err::<P, F>(
                            buf,
                            self.buf_reader.file_offset(),
                            // &self.record_data,
                            self.record_data.pos(),
                            *self.record_data.record_offset(),
                            self.record_data.read_opts(),
                            self.record_data.mut_read_opts(),
                            Some(self.search_pos),
                            // &self.record_data.header,
                        ))
                    }
                }
            }
        
            // This function is only to be called if the buffer needs to be adjusted,
            // either by making room or by growing
            #[inline]
            fn adjust_buffer(&mut self) -> error::Result<()> {
                // TODO: dangerous, may not be correct, data_start may slow down
                let start = self
                    .record_data
                    .pos()
                    .start::<F::Parser>(self.search_pos, self.record_data.read_opts(), self.record_data.mut_read_opts());
                if start == 0 {
                    // TODO: larger threshold for efficiency?
                    // first record already incomplete -> buffer too small
                    self.grow()?;
                } else {
                    // not the first record -> buffer may be big enough
                    self.make_room()?;
                }
                Ok(())
            }
        
            #[inline]
            fn grow(&mut self) -> error::Result<()> {
                if let Some(limit) = self.buf_reader.grow_limited() {
                    return Err(error::Error::new(error::ErrorKind::BufferLimit(limit)));
                }
                self.fill_buf()?;
                //println!("grow {:?}", self.buf_reader.capacity());
                Ok(())
            }
        
            #[inline]
            fn make_room(&mut self) -> error::Result<()> {
                // move incomplete bytes to start of buffer and retry
                let offset = self
                    .record_data
                    .pos
                    .0
                    .start::<F::Parser>(self.search_pos, self.record_data.read_opts(), self.record_data.mut_read_opts());
                self.buf_reader.make_room(offset);
                // TODO: eventually use self.search_pos
                self.record_data.pos_mut().apply_offset(0 - offset as isize);
                self.search_pos.cursor -= offset;
                self.fill_buf()?;
            // println!("make room {:?} {:?}", self.buf_reader.buffer().len(), offset);
            // println!("new buf {:?}", std::str::from_utf8(self.buf_reader.buffer()));
                Ok(())
            }
        
            /// Fills the internal buffer (if not already full)
            /// Performance tests have shown that it is fastest to not inline this.
            /// There are several calls to this function within this module.
            #[inline(never)]
            fn fill_buf(&mut self) -> error::Result<usize> {
                self.buf_reader
                    .fill_buf()
                    .map_err(|e| error::Error::new(error::ErrorKind::Io(e)))
            }
        }
    };
}