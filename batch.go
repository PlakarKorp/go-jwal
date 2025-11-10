package jwal

import (
	"encoding/binary"
	"hash/crc32"
	"io"
)

type Batch struct {
	recs [][]byte
}

func (b *Batch) Add(p []byte) {
	b.recs = append(b.recs, p)
}

func (b *Batch) Reset() {
	b.recs = b.recs[:0]
}

func (b *Batch) Len() int {
	return len(b.recs)
}

func (b *Batch) Records() [][]byte {
	return b.recs
}

func (l *Log) AppendBatch(records ...[]byte) (first, last uint64, err error) {
	if len(records) == 0 {
		l.mu.Lock()
		defer l.mu.Unlock()
		return l.count.Load(), l.count.Load(), nil
	}
	return l.appendBatch(records)
}

func (l *Log) WriteBatch(b *Batch) (first, last uint64, err error) {
	if b == nil || len(b.recs) == 0 {
		l.mu.Lock()
		defer l.mu.Unlock()
		return l.count.Load(), l.count.Load(), nil
	}
	return l.appendBatch(b.recs)
}

func (l *Log) appendBatch(records [][]byte) (first, last uint64, err error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	var hdr [recordHdrSize]byte

	hdrOff, err := l.fp.Seek(0, io.SeekEnd)
	if err != nil {
		return 0, 0, err
	}

	// Write all headers + payloads
	offs := make([]int64, 0, len(records)) // for full index
	off := hdrOff

	for _, p := range records {
		payload, storedLen, ulen := l.preparePayloadLocked(p)

		// header
		binary.LittleEndian.PutUint64(hdr[0:8], storedLen)
		binary.LittleEndian.PutUint32(hdr[8:12], crc32.ChecksumIEEE(payload))
		binary.LittleEndian.PutUint32(hdr[12:16], ulen)

		if _, err := l.w.Write(hdr[:]); err != nil {
			return 0, 0, err
		}
		if _, err := l.w.Write(payload); err != nil {
			return 0, 0, err
		}

		// advance expected file offset for index math
		offs = append(offs, off+recordHdrSize) // payload start
		off += recordHdrSize + int64(storedLen)
	}

	if err := l.w.Flush(); err != nil {
		return 0, 0, err
	}
	if !l.noSync {
		if err := l.fp.Sync(); err != nil {
			return 0, 0, err
		}
	}

	old := l.count.Load()
	first = old + 1
	nRecs := uint64(len(records))

	if l.retainIndex {
		if l.checkpointInterval == 1 {
			l.idx = append(l.idx, offs...)
		} else {
			// sparse: add checkpoints for records 1, 1+K, ...
			k := uint64(l.checkpointInterval)
			off := hdrOff
			for i := range records {
				next := old + uint64(i) + 1
				if (next-1)%k == 0 {
					l.idx = append(l.idx, off) // header offset
				}
				// recompute storedLen cheaply from the header we wrote:
				// but we already tracked it via offs/advance above:
				// off += headerSize + storedLen
				// We can derive it from consecutive offs:
				if i+1 < len(offs) {
					// next header offset = payloadStartNext - headerSize
					nextHdr := offs[i+1] - recordHdrSize
					off = nextHdr
				} else {
					// final position already in 'off' at end of loop above
				}
			}
		}
	}

	l.count.Add(nRecs)
	last = l.count.Load()
	return first, last, nil
}
