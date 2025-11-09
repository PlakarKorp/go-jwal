package jwal

import (
	"bufio"
	"encoding/binary"
	"hash/crc32"
	"io"
	"os"
	"sync"
)

// Record layout (little-endian):
// [0..7]   uint64 len
// [8..11]  uint32 crc32(payload)
// [12..15] uint32 reserved (0)
// [16..]   payload
const headerSize = 16

type Options struct {
	NoSync             bool // if true, Append won't fsync; call Sync manually
	BufferSize         int  // bufio size; default 256 KiB if <=0
	RetainIndex        bool // if true, keep an index (full or sparse)
	CheckpointInterval uint // when RetainIndex=true:
	//   1 => keep every record offset (full index, ~8B/record)
	//   >1 => keep every K-th record header offset (sparse)
	//   0 => defaults to 4096 (sparse)
}

type Log struct {
	mu sync.Mutex
	fp *os.File
	w  *bufio.Writer

	hdr [headerSize]byte
	idx []int64

	noSync             bool
	retainIndex        bool
	checkpointInterval uint
	count              uint64 // total records, 1-based indexing
}

func Open(path string, opts *Options) (*Log, error) {
	if opts == nil {
		opts = &Options{}
	}
	fp, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600)
	if err != nil {
		return nil, err
	}

	bufsz := opts.BufferSize
	if bufsz <= 0 {
		bufsz = 256 << 10
	}

	l := &Log{
		fp:                 fp,
		w:                  bufio.NewWriterSize(fp, bufsz),
		noSync:             opts.NoSync,
		retainIndex:        opts.RetainIndex,
		checkpointInterval: opts.CheckpointInterval,
	}
	if l.retainIndex {
		if l.checkpointInterval == 0 {
			l.checkpointInterval = 4096
		}
		if l.checkpointInterval < 1 {
			l.checkpointInterval = 1
		}
	}

	if err := l.scanAndRecover(); err != nil {
		_ = fp.Close()
		return nil, err
	}
	return l, nil
}

func (l *Log) scanAndRecover() error {
	var off int64 // header offset of current record
	var buf []byte

	fi, err := l.fp.Stat()
	if err != nil {
		return err
	}
	size := fi.Size()

	for off+headerSize <= size {
		if _, err := l.fp.ReadAt(l.hdr[:], off); err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
		n := binary.LittleEndian.Uint64(l.hdr[0:8])
		end := off + headerSize + int64(n)
		if end > size {
			return l.fp.Truncate(off)
		}
		want := binary.LittleEndian.Uint32(l.hdr[8:12])

		if len(buf) < int(n) {
			buf = make([]byte, int(n))
		}
		if _, err := l.fp.ReadAt(buf[:int(n)], off+headerSize); err != nil {
			if err == io.EOF {
				return l.fp.Truncate(off)
			}
			return err
		}
		got := crc32.ChecksumIEEE(buf[:int(n)])
		if got != want {
			return l.fp.Truncate(off)
		}

		// good record → maybe index it
		if l.retainIndex {
			if l.checkpointInterval == 1 {
				// full index stores DATA-start offsets
				l.idx = append(l.idx, off+headerSize)
			} else {
				recNum := l.count + 1
				if (recNum-1)%uint64(l.checkpointInterval) == 0 {
					// sparse stores HEADER offsets at checkpoints
					l.idx = append(l.idx, off)
				}
			}
		}
		l.count++
		off = end
	}
	_, err = l.fp.Seek(0, io.SeekEnd)
	return err
}

func (l *Log) Append(data []byte) (uint64, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	hdrOff, err := l.fp.Seek(0, io.SeekEnd)
	if err != nil {
		return 0, err
	}
	n := uint64(len(data))
	binary.LittleEndian.PutUint64(l.hdr[0:8], n)
	binary.LittleEndian.PutUint32(l.hdr[8:12], crc32.ChecksumIEEE(data))

	if _, err := l.w.Write(l.hdr[:]); err != nil {
		return 0, err
	}
	if _, err := l.w.Write(data); err != nil {
		return 0, err
	}
	if err := l.w.Flush(); err != nil {
		return 0, err
	}
	if !l.noSync {
		if err := l.fp.Sync(); err != nil {
			return 0, err
		}
	}

	// update index
	if l.retainIndex {
		if l.checkpointInterval == 1 {
			curEnd, _ := l.fp.Seek(0, io.SeekCurrent)
			l.idx = append(l.idx, curEnd-int64(len(data))) // DATA-start offset
		} else {
			next := l.count + 1
			if (next-1)%uint64(l.checkpointInterval) == 0 {
				l.idx = append(l.idx, hdrOff) // HEADER offset checkpoint
			}
		}
	}

	l.count++
	return l.count, nil
}

type Batch struct {
	recs [][]byte
}

func (b *Batch) Add(p []byte)      { b.recs = append(b.recs, p) }
func (b *Batch) Reset()            { b.recs = b.recs[:0] }
func (b *Batch) Len() int          { return len(b.recs) }
func (b *Batch) Records() [][]byte { return b.recs }
func (l *Log) AppendBatch(records ...[]byte) (first, last uint64, err error) {
	if len(records) == 0 {
		l.mu.Lock()
		defer l.mu.Unlock()
		return l.count, l.count, nil
	}
	return l.appendBatch(records)
}
func (l *Log) WriteBatch(b *Batch) (first, last uint64, err error) {
	if b == nil || len(b.recs) == 0 {
		l.mu.Lock()
		defer l.mu.Unlock()
		return l.count, l.count, nil
	}
	return l.appendBatch(b.recs)
}

func (l *Log) appendBatch(records [][]byte) (first, last uint64, err error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	hdrOff, err := l.fp.Seek(0, io.SeekEnd)
	if err != nil {
		return 0, 0, err
	}

	for _, p := range records {
		binary.LittleEndian.PutUint64(l.hdr[0:8], uint64(len(p)))
		binary.LittleEndian.PutUint32(l.hdr[8:12], crc32.ChecksumIEEE(p))
		if _, err := l.w.Write(l.hdr[:]); err != nil {
			return 0, 0, err
		}
		if _, err := l.w.Write(p); err != nil {
			return 0, 0, err
		}
	}
	if err := l.w.Flush(); err != nil {
		return 0, 0, err
	}
	if !l.noSync {
		if err := l.fp.Sync(); err != nil {
			return 0, 0, err
		}
	}

	old := l.count
	first = old + 1
	nRecs := uint64(len(records))

	if l.retainIndex {
		if l.checkpointInterval == 1 {
			// full index: append all DATA-start offsets
			off := hdrOff
			for _, p := range records {
				l.idx = append(l.idx, off+headerSize)
				off += headerSize + int64(len(p))
			}
		} else {
			// sparse: add checkpoints for records 1, 1+K, ...
			k := uint64(l.checkpointInterval)
			off := hdrOff
			for i, p := range records {
				next := old + uint64(i) + 1
				if (next-1)%k == 0 {
					l.idx = append(l.idx, off) // HEADER offset
				}
				off += headerSize + int64(len(p))
			}
		}
	}

	l.count += nRecs
	last = l.count
	return first, last, nil
}

func (l *Log) Read(index uint64) ([]byte, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	if index == 0 || index > l.count {
		return nil, io.EOF
	}
	dataOff, err := l.locateDataOffsetLocked(index)
	if err != nil {
		return nil, err
	}
	if _, err := l.fp.ReadAt(l.hdr[:], dataOff-headerSize); err != nil {
		return nil, err
	}
	n := int(binary.LittleEndian.Uint64(l.hdr[0:8]))
	buf := make([]byte, n)
	if _, err := l.fp.ReadAt(buf, dataOff); err != nil {
		return nil, err
	}
	if crc32.ChecksumIEEE(buf) != binary.LittleEndian.Uint32(l.hdr[8:12]) {
		return nil, io.ErrUnexpectedEOF
	}
	return buf, nil
}

func (l *Log) ReadInto(index uint64, dst []byte) ([]byte, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	if index == 0 || index > l.count {
		return nil, io.EOF
	}
	dataOff, err := l.locateDataOffsetLocked(index)
	if err != nil {
		return nil, err
	}
	if _, err := l.fp.ReadAt(l.hdr[:], dataOff-headerSize); err != nil {
		return nil, err
	}
	n := int(binary.LittleEndian.Uint64(l.hdr[0:8]))
	if cap(dst) < n {
		dst = make([]byte, n)
	} else {
		dst = dst[:n]
	}
	if _, err := l.fp.ReadAt(dst, dataOff); err != nil {
		return nil, err
	}
	if crc32.ChecksumIEEE(dst) != binary.LittleEndian.Uint32(l.hdr[8:12]) {
		return nil, io.ErrUnexpectedEOF
	}
	return dst, nil
}

func (l *Log) LastIndex() uint64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.count
}

func (l *Log) TruncateBack(index uint64) error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if index > l.count {
		return io.EOF
	}

	var newSize int64
	if index == 0 {
		newSize = 0
	} else {
		dataOff, err := l.locateDataOffsetLocked(index)
		if err != nil {
			return err
		}
		if _, err := l.fp.ReadAt(l.hdr[:], dataOff-headerSize); err != nil {
			return err
		}
		n := int64(binary.LittleEndian.Uint64(l.hdr[0:8]))
		newSize = (dataOff - headerSize) + headerSize + n
	}

	if err := l.w.Flush(); err != nil {
		return err
	}
	if err := l.fp.Truncate(newSize); err != nil {
		return err
	}
	if _, err := l.fp.Seek(newSize, io.SeekStart); err != nil {
		return err
	}

	// fix in-memory index
	if l.retainIndex {
		if l.checkpointInterval == 1 {
			if index < uint64(len(l.idx)) {
				l.idx = l.idx[:index]
			}
		} else {
			cpCount := 0
			if index > 0 {
				cpCount = int((index-1)/uint64(l.checkpointInterval) + 1)
			}
			if cpCount < len(l.idx) {
				l.idx = l.idx[:cpCount]
			}
		}
	}
	l.count = index
	return nil
}

func (l *Log) Sync() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.flushAndSyncLocked()
}

func (l *Log) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	if err := l.flushAndSyncLocked(); err != nil {
		_ = l.fp.Close()
		return err
	}
	return l.fp.Close()
}

func (l *Log) flushAndSyncLocked() error {
	if err := l.w.Flush(); err != nil {
		return err
	}
	if l.noSync {
		return nil
	}
	return l.fp.Sync()
}

func (l *Log) locateDataOffsetLocked(index uint64) (int64, error) {
	if l.retainIndex {
		// full index, direct lookup
		if l.checkpointInterval == 1 {
			return l.idx[index-1], nil
		}

		// sparse ? jump to nearest checkpoint <= index, then scan forward
		k := uint64(l.checkpointInterval)
		cp := (index - 1) / k

		var hdrOff int64
		var startIdx uint64

		if len(l.idx) > 0 {
			if int(cp) < len(l.idx) {
				hdrOff = l.idx[cp]
				startIdx = cp*k + 1
			} else {
				// beyond last checkpoint (recent data, no checkpoint yet)
				hdrOff = l.idx[len(l.idx)-1]
				startIdx = (uint64(len(l.idx))-1)*k + 1
			}
		} else {
			// no checkpoints yet ? start from beginning
			hdrOff = 0
			startIdx = 1
		}

		off := hdrOff
		for cur := startIdx; cur < index; cur++ {
			if _, err := l.fp.ReadAt(l.hdr[:], off); err != nil {
				return 0, err
			}
			n := int64(binary.LittleEndian.Uint64(l.hdr[0:8]))
			off += headerSize + n
		}
		return off + headerSize, nil
	}

	// no index ? linear scan from beginning
	var off int64
	for cur := uint64(1); cur <= index; cur++ {
		if _, err := l.fp.ReadAt(l.hdr[:], off); err != nil {
			return 0, err
		}
		n := int64(binary.LittleEndian.Uint64(l.hdr[0:8]))
		if cur == index {
			return off + headerSize, nil
		}
		off += headerSize + n
	}
	return 0, io.EOF
}
