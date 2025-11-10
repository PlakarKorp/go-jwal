package jwal

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func benchPath(b *testing.B) string {
	return filepath.Join(b.TempDir(), "bench.jwal")
}

func payloadOf(n int) []byte {
	p := make([]byte, n)
	_, _ = rand.Read(p)
	return p
}

// --- APPEND (NoSync=true) ----------------------------------------------------

func BenchmarkAppend_NoSync(b *testing.B) {
	sizes := []int{16, 128, 1024, 4096, 1 << 16}
	for _, sz := range sizes {
		b.Run(bytesSize(sz), func(b *testing.B) {
			path := benchPath(b)
			l, err := Open(path, &Options{
				NoSync:             true,
				BufferSize:         256 << 10,
				RetainIndex:        false, // fastest write path
				CheckpointInterval: 0,
			})
			if err != nil {
				b.Fatalf("Open: %v", err)
			}
			defer l.Close()

			data := payloadOf(sz)
			b.SetBytes(int64(sz))
			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				if _, err := l.Append(data); err != nil {
					b.Fatalf("Append: %v", err)
				}
			}
			b.StopTimer()
		})
	}
}

// --- APPEND (durable: fsync each Append) -------------------------------------

func BenchmarkAppend_SyncEach(b *testing.B) {
	sizes := []int{16, 128, 1024, 4096}
	for _, sz := range sizes {
		b.Run(bytesSize(sz), func(b *testing.B) {
			path := benchPath(b)
			l, err := Open(path, &Options{
				NoSync:             false, // fsync per append
				BufferSize:         256 << 10,
				RetainIndex:        false,
				CheckpointInterval: 0,
			})
			if err != nil {
				b.Fatalf("Open: %v", err)
			}
			defer l.Close()

			data := payloadOf(sz)
			b.SetBytes(int64(sz))
			b.ReportAllocs()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				if _, err := l.Append(data); err != nil {
					b.Fatalf("Append: %v", err)
				}
			}
			b.StopTimer()
		})
	}
}

// --- APPEND BATCH (NoSync=true) ----------------------------------------------

func BenchmarkAppendBatch_NoSync(b *testing.B) {
	sizes := []int{256, 1024, 4096}
	batchLens := []int{4, 16, 64}
	for _, sz := range sizes {
		for _, bl := range batchLens {
			name := bytesSize(sz) + "_batch" + itoa(bl)
			b.Run(name, func(b *testing.B) {
				path := benchPath(b)
				l, err := Open(path, &Options{
					NoSync:             true,
					BufferSize:         256 << 10,
					RetainIndex:        false,
					CheckpointInterval: 0,
				})
				if err != nil {
					b.Fatalf("Open: %v", err)
				}
				defer l.Close()

				data := payloadOf(sz)
				records := make([][]byte, bl)
				for i := 0; i < bl; i++ {
					records[i] = data
				}

				b.SetBytes(int64(sz * bl))
				b.ReportAllocs()
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					if _, _, err := l.AppendBatch(records...); err != nil {
						b.Fatalf("AppendBatch: %v", err)
					}
				}
				b.StopTimer()
			})
		}
	}
}

// --- APPEND BATCH (durable: single fsync per batch) --------------------------

func BenchmarkAppendBatch_SyncEach(b *testing.B) {
	sz := 1024
	bl := 32
	path := benchPath(b)
	l, err := Open(path, &Options{
		NoSync:             false, // fsync once per batch
		BufferSize:         256 << 10,
		RetainIndex:        false,
		CheckpointInterval: 0,
	})
	if err != nil {
		b.Fatalf("Open: %v", err)
	}
	defer l.Close()

	data := payloadOf(sz)
	records := make([][]byte, bl)
	for i := 0; i < bl; i++ {
		records[i] = data
	}

	b.SetBytes(int64(sz * bl))
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if _, _, err := l.AppendBatch(records...); err != nil {
			b.Fatalf("AppendBatch: %v", err)
		}
	}
	b.StopTimer()
}

// --- PARALLEL APPEND (NoSync=true) -------------------------------------------

func BenchmarkAppend_Parallel_NoSync(b *testing.B) {
	const sz = 1024
	path := benchPath(b)
	l, err := Open(path, &Options{
		NoSync:             true,
		BufferSize:         256 << 10,
		RetainIndex:        false,
		CheckpointInterval: 0,
	})
	if err != nil {
		b.Fatalf("Open: %v", err)
	}
	defer l.Close()

	data := payloadOf(sz)
	b.SetBytes(int64(sz))
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if _, err := l.Append(data); err != nil {
				b.Fatalf("Append: %v", err)
			}
		}
	})
	b.StopTimer()
}

// --- READ SEQUENTIAL (from disk filled ahead of time) ------------------------

func BenchmarkReadSequential(b *testing.B) {
	sizes := []int{16, 128, 1024, 4096, 1 << 16}
	for _, sz := range sizes {
		b.Run(bytesSize(sz), func(b *testing.B) {
			path := benchPath(b)

			// prefill with ~max(b.N, 16k) records to avoid skew
			prefill := max(b.N, 1<<14)
			data := payloadOf(sz)
			{
				l, err := Open(path, &Options{
					NoSync:             true,
					BufferSize:         256 << 10,
					RetainIndex:        true, // allow either
					CheckpointInterval: 4096, // sparse by default; set to 1 for O(1)
				})
				if err != nil {
					b.Fatalf("Open(prefill): %v", err)
				}
				for i := 0; i < prefill; i++ {
					if _, err := l.Append(data); err != nil {
						b.Fatalf("Append(prefill): %v", err)
					}
				}
				if err := l.Close(); err != nil {
					b.Fatalf("Close(prefill): %v", err)
				}
			}

			// reopen for reads
			l, err := Open(path, &Options{
				NoSync:             true,
				BufferSize:         256 << 10,
				RetainIndex:        true,
				CheckpointInterval: 4096,
			})
			if err != nil {
				b.Fatalf("Open(read): %v", err)
			}
			defer l.Close()

			b.SetBytes(int64(sz))
			b.ReportAllocs()
			b.ResetTimer()

			var idx uint64 = 1
			last := l.LastIndex()
			for i := 0; i < b.N; i++ {
				if idx > last {
					idx = 1
				}
				buf, err := l.Read(idx)
				if err != nil {
					b.Fatalf("Read(%d): %v", idx, err)
				}
				if len(buf) != sz {
					b.Fatalf("size mismatch: got %d want %d", len(buf), sz)
				}
				idx++
			}
			b.StopTimer()
		})
	}
}

// --- READ SEQUENTIAL using ReadInto (zero-alloc hot path) --------------------

func BenchmarkReadSequential_Into(b *testing.B) {
	const sz = 4096
	path := benchPath(b)

	data := payloadOf(sz)
	// prefill
	{
		l, err := Open(path, &Options{
			NoSync:             true,
			BufferSize:         256 << 10,
			RetainIndex:        true,
			CheckpointInterval: 1, // full index for pure O(1)
		})
		if err != nil {
			b.Fatalf("Open(prefill): %v", err)
		}
		for i := 0; i < max(b.N, 1<<14); i++ {
			if _, err := l.Append(data); err != nil {
				b.Fatalf("Append(prefill): %v", err)
			}
		}
		_ = l.Close()
	}

	l, err := Open(path, &Options{
		NoSync:             true,
		BufferSize:         256 << 10,
		RetainIndex:        true,
		CheckpointInterval: 1,
	})
	if err != nil {
		b.Fatalf("Open(read): %v", err)
	}
	defer l.Close()

	dst := make([]byte, 0, sz)
	b.SetBytes(int64(sz))
	b.ReportAllocs()
	b.ResetTimer()

	var idx uint64 = 1
	last := l.LastIndex()
	for i := 0; i < b.N; i++ {
		if idx > last {
			idx = 1
		}
		var err error
		dst, err = l.ReadInto(idx, dst[:0])
		if err != nil {
			b.Fatalf("ReadInto(%d): %v", idx, err)
		}
		if len(dst) != sz {
			b.Fatalf("size mismatch: got %d", len(dst))
		}
		idx++
	}
	b.StopTimer()
}

// --- READ STREAMING ITERATION (no index, linear scan) ------------------------

func BenchmarkReadStreaming_NoIndex(b *testing.B) {
	const sz = 2048
	path := benchPath(b)

	data := payloadOf(sz)
	// prefill without index to test pure scan
	{
		l, err := Open(path, &Options{
			NoSync:             true,
			BufferSize:         256 << 10,
			RetainIndex:        false,
			CheckpointInterval: 0,
		})
		if err != nil {
			b.Fatalf("Open(prefill): %v", err)
		}
		for i := 0; i < max(b.N, 1<<14); i++ {
			if _, err := l.Append(data); err != nil {
				b.Fatalf("Append(prefill): %v", err)
			}
		}
		_ = l.Close()
	}

	// streaming scan by increasing index each time (forces header walks)
	l, err := Open(path, &Options{
		NoSync:             true,
		BufferSize:         256 << 10,
		RetainIndex:        false,
		CheckpointInterval: 0,
	})
	if err != nil {
		b.Fatalf("Open(read): %v", err)
	}
	defer l.Close()

	b.SetBytes(int64(sz))
	b.ReportAllocs()
	b.ResetTimer()

	var idx uint64 = 1
	last := l.LastIndex()
	for i := 0; i < b.N; i++ {
		if idx > last {
			idx = 1
		}
		buf, err := l.Read(idx)
		if err != nil && err != io.EOF {
			b.Fatalf("Read(%d): %v", idx, err)
		}
		if len(buf) != sz {
			b.Fatalf("size mismatch: got %d want %d", len(buf), sz)
		}
		idx++
	}
	b.StopTimer()
}

// --- helpers -----------------------------------------------------------------

func bytesSize(n int) string {
	switch {
	case n >= 1<<20:
		return itoa(n>>20) + "MiB"
	case n >= 1<<10:
		return itoa(n>>10) + "KiB"
	default:
		return itoa(n) + "B"
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var buf [20]byte
	pos := len(buf)
	for i > 0 {
		pos--
		buf[pos] = byte('0' + i%10)
		i /= 10
	}
	return string(buf[pos:])
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// test helpers

type openOpt struct {
	comp          string
	retain        bool
	checkptK      uint
	noSync        bool
	bufSize       int
	deleteOnClose bool
}

func mustOpen(t *testing.T, p string, o openOpt) *Log {
	t.Helper()
	l, err := Open(p, &Options{
		NoSync:             o.noSync,
		BufferSize:         o.bufSize,
		RetainIndex:        o.retain,
		CheckpointInterval: o.checkptK,
		Compression:        o.comp,
		DeleteOnClose:      o.deleteOnClose,
	})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return l
}

func tmpPath(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	return filepath.Join(dir, "test.jwal")
}

func payload(n int) []byte {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return b
}

func readFile(t *testing.T, p string) []byte {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	return b
}

func fileSize(t *testing.T, p string) int64 {
	t.Helper()
	fi, err := os.Stat(p)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	return fi.Size()
}

// ---- TESTS ----

func TestHeaderAndEmptyFile(t *testing.T) {
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "none"})
	defer l.Close()

	if got, want := fileSize(t, p), int64(walHdrSize); got != want {
		t.Fatalf("size=%d want WAL header %d", got, want)
	}
	if got := l.LastIndex(); got != 0 {
		t.Fatalf("LastIndex=%d want 0", got)
	}
}

func TestAppendRead_NoCompression(t *testing.T) {
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "none"})
	defer l.Close()

	in := []byte("hello world")
	idx, err := l.Append(in)
	if err != nil {
		t.Fatalf("Append: %v", err)
	}
	if idx != 1 {
		t.Fatalf("idx=%d want 1", idx)
	}

	out, err := l.Read(1)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if !bytes.Equal(out, in) {
		t.Fatalf("payload mismatch")
	}

	// Also test ReadInto
	buf := make([]byte, 0, len(in))
	out2, err := l.ReadInto(1, buf)
	if err != nil {
		t.Fatalf("ReadInto: %v", err)
	}
	if !bytes.Equal(out2, in) {
		t.Fatalf("ReadInto mismatch")
	}

	// sanity on file size: header + record header + payload
	sz := fileSize(t, p)
	if sz < int64(walHdrSize+recordHdrSize+len(in)) {
		t.Fatalf("unexpected small file size: %d", sz)
	}
}

func TestAppendRead_Snappy(t *testing.T) {
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "snappy"})
	defer l.Close()

	in := payload(64 << 10) // 64 KiB
	idx, err := l.Append(in)
	if err != nil {
		t.Fatalf("Append: %v", err)
	}
	if idx != 1 {
		t.Fatalf("idx=%d want 1", idx)
	}

	out, err := l.Read(1)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if !bytes.Equal(out, in) {
		t.Fatalf("payload mismatch")
	}

	buf := make([]byte, 0, len(in))
	out2, err := l.ReadInto(1, buf)
	if err != nil {
		t.Fatalf("ReadInto: %v", err)
	}
	if !bytes.Equal(out2, in) {
		t.Fatalf("ReadInto mismatch")
	}
}

func TestAppendBatch_FullIndex(t *testing.T) {
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "none", retain: true, checkptK: 1})
	defer l.Close()

	var recs [][]byte
	for i := 0; i < 10; i++ {
		recs = append(recs, []byte{byte(i), byte(i + 1), byte(i + 2)})
	}

	first, last, err := l.AppendBatch(recs...)
	if err != nil {
		t.Fatalf("AppendBatch: %v", err)
	}
	if first != 1 || last != 10 {
		t.Fatalf("range got [%d,%d] want [1,10]", first, last)
	}

	// verify reads via full index
	for i := 1; i <= 10; i++ {
		out, err := l.Read(uint64(i))
		if err != nil {
			t.Fatalf("Read(%d): %v", i, err)
		}
		if !bytes.Equal(out, recs[i-1]) {
			t.Fatalf("rec %d mismatch", i)
		}
	}
}

func TestAppendBatch_SparseIndex(t *testing.T) {
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "snappy", retain: true, checkptK: 4})
	defer l.Close()

	var recs [][]byte
	for i := 0; i < 25; i++ {
		recs = append(recs, payload(200+i))
	}

	first, last, err := l.AppendBatch(recs...)
	if err != nil {
		t.Fatalf("AppendBatch: %v", err)
	}
	if first != 1 || last != 25 {
		t.Fatalf("range got [%d,%d] want [1,25]", first, last)
	}

	// verify reads on a few positions (forces sparse checkpoint + scan ahead)
	for _, i := range []int{1, 4, 5, 8, 16, 17, 24, 25} {
		out, err := l.Read(uint64(i))
		if err != nil {
			t.Fatalf("Read(%d): %v", i, err)
		}
		if !bytes.Equal(out, recs[i-1]) {
			t.Fatalf("rec %d mismatch", i)
		}
	}
}

func TestReopenAndTailRecovery_TornTail(t *testing.T) {
	p := tmpPath(t)
	{
		l := mustOpen(t, p, openOpt{comp: "none"})
		defer l.Close()

		if _, err := l.Append([]byte("ok-1")); err != nil {
			t.Fatalf("append: %v", err)
		}
		if _, err := l.Append([]byte("ok-2")); err != nil {
			t.Fatalf("append: %v", err)
		}
	}
	// Simulate torn tail by truncating in the middle of last record payload
	// layout: walHdr | recHdr | "ok-1" | recHdr | "ok-2"
	// we truncate a few bytes from EOF
	fs := fileSize(t, p)
	if err := os.Truncate(p, fs-2); err != nil { // damage
		t.Fatalf("Truncate: %v", err)
	}

	// Reopen should truncate back to end of the last valid record (i.e., drop "ok-2")
	l2 := mustOpen(t, p, openOpt{comp: "none"})
	defer l2.Close()

	if got := l2.LastIndex(); got != 1 {
		t.Fatalf("LastIndex after recovery=%d want 1", got)
	}
	out, err := l2.Read(1)
	if err != nil {
		t.Fatalf("Read(1): %v", err)
	}
	if string(out) != "ok-1" {
		t.Fatalf("record content mismatch after recovery")
	}
}

func TestReopenAndTailRecovery_CRC(t *testing.T) {
	p := tmpPath(t)
	{
		l := mustOpen(t, p, openOpt{comp: "none"})
		defer l.Close()

		if _, err := l.Append([]byte("keep")); err != nil {
			t.Fatalf("append: %v", err)
		}
		if _, err := l.Append([]byte("break-this")); err != nil {
			t.Fatalf("append: %v", err)
		}
	}

	// Flip a byte inside the second payload to make CRC fail.
	f, err := os.OpenFile(p, os.O_RDWR, 0)
	if err != nil {
		t.Fatalf("openfile: %v", err)
	}
	defer f.Close()

	// Seek to the second record payload start to corrupt it.
	// We know layout: walHdr, recHdr, "keep", recHdr, "break-this"
	// Walk manually: read first header to learn storedLen, then skip payload,
	// then at second header + recordHdrSize is payload start.
	// (This test keeps assumptions local and avoids touching unexported bits.)

	// First header at walHdrSize
	hdr1 := make([]byte, recordHdrSize)
	if _, err := f.ReadAt(hdr1, walHdrSize); err != nil {
		t.Fatalf("read hdr1: %v", err)
	}
	sz1 := int64(binary.LittleEndian.Uint64(hdr1[0:8]))
	off2hdr := int64(walHdrSize) + int64(recordHdrSize) + sz1
	off2payload := off2hdr + int64(recordHdrSize)

	// Corrupt 1 byte in second payload
	b := make([]byte, 1)
	if _, err := f.ReadAt(b, off2payload); err != nil {
		t.Fatalf("read payload byte: %v", err)
	}
	b[0] ^= 0xFF
	if _, err := f.WriteAt(b, off2payload); err != nil {
		t.Fatalf("write corrupt: %v", err)
	}

	// Reopen: the scanner should truncate at the *start* of second header.
	l2 := mustOpen(t, p, openOpt{comp: "none"})
	defer l2.Close()

	if got := l2.LastIndex(); got != 1 {
		t.Fatalf("LastIndex after CRC recovery=%d want 1", got)
	}
	out, err := l2.Read(1)
	if err != nil {
		t.Fatalf("Read(1): %v", err)
	}
	if string(out) != "keep" {
		t.Fatalf("record content mismatch after CRC recovery")
	}
}

func TestTruncateBack_PreservesHeader(t *testing.T) {
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "snappy"})
	defer l.Close()

	for i := 0; i < 5; i++ {
		if _, err := l.Append(payload(128)); err != nil {
			t.Fatalf("append: %v", err)
		}
	}
	if err := l.TruncateBack(2); err != nil {
		t.Fatalf("TruncateBack: %v", err)
	}

	if got := l.LastIndex(); got != 2 {
		t.Fatalf("LastIndex=%d want 2", got)
	}
	// file must be at least WAL header
	if sz := fileSize(t, p); sz < walHdrSize {
		t.Fatalf("file too small after truncate: %d", sz)
	}

	// Truncate to 0 -> keep WAL header
	if err := l.TruncateBack(0); err != nil {
		t.Fatalf("TruncateBack(0): %v", err)
	}
	if got := l.LastIndex(); got != 0 {
		t.Fatalf("LastIndex=%d want 0", got)
	}
	if sz := fileSize(t, p); sz != walHdrSize {
		t.Fatalf("file size=%d want exactly walHdrSize=%d", sz, walHdrSize)
	}
}

func TestReadBounds(t *testing.T) {
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "none"})
	defer l.Close()

	if _, err := l.Read(1); !errors.Is(err, io.EOF) {
		t.Fatalf("expected EOF on empty log, got %v", err)
	}
	if _, err := l.Append([]byte("x")); err != nil {
		t.Fatalf("append: %v", err)
	}
	if _, err := l.Read(2); !errors.Is(err, io.EOF) {
		t.Fatalf("expected EOF on out-of-range, got %v", err)
	}
}

func TestSyncOption(t *testing.T) {
	p := tmpPath(t)
	// Open with NoSync; still should write and be reopenable.
	l := mustOpen(t, p, openOpt{comp: "snappy", noSync: true})
	if _, err := l.Append([]byte("nsync-1")); err != nil {
		t.Fatalf("append: %v", err)
	}
	if err := l.Sync(); err != nil {
		t.Fatalf("Sync: %v", err)
	}
	if err := l.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	l2 := mustOpen(t, p, openOpt{comp: "snappy", noSync: true})
	defer l2.Close()
	out, err := l2.Read(1)
	if err != nil {
		t.Fatalf("Read(1): %v", err)
	}
	if string(out) != "nsync-1" {
		t.Fatalf("payload mismatch")
	}
}

func TestIterator_Basic(t *testing.T) {
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "none"})
	defer l.Close()

	expect := [][]byte{
		[]byte("a"),
		[]byte("bb"),
		[]byte("ccc"),
	}
	for _, x := range expect {
		if _, err := l.Append(x); err != nil {
			t.Fatalf("append: %v", err)
		}
	}

	// NOTE: As currently implemented, Iter(from==1) sets hdrOff=0.
	// With a WAL header present, the first record header starts at walHdrSize.
	// This test expects correct behavior (start at walHdrSize). If it fails,
	// the fix is to set hdrOff = dataBase when from == 1.
	it, err := l.Iter(1)
	if err != nil {
		t.Fatalf("Iter: %v", err)
	}

	var got [][]byte
	for {
		b, idx, err := it.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("iter next: %v", err)
		}
		if idx < 1 || idx > uint64(len(expect)) {
			t.Fatalf("unexpected idx %d", idx)
		}
		got = append(got, b)
	}
	if len(got) != len(expect) {
		t.Fatalf("iter len=%d want %d", len(got), len(expect))
	}
	for i := range expect {
		if !bytes.Equal(expect[i], got[i]) {
			t.Fatalf("iter rec %d mismatch", i+1)
		}
	}
}

func TestDataOffsetAPI(t *testing.T) {
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "none"})
	defer l.Close()

	msg := []byte("hello")
	if _, err := l.Append(msg); err != nil {
		t.Fatalf("append: %v", err)
	}

	off, err := l.DataOffset(1)
	if err != nil {
		t.Fatalf("DataOffset: %v", err)
	}
	// Read raw header then payload using offsets, verify CRC and bytes.
	h := make([]byte, recordHdrSize)
	if _, err := l.ReadAt(h, off-recordHdrSize); err != nil {
		t.Fatalf("ReadAt header: %v", err)
	}
	storedLen := int(binary.LittleEndian.Uint64(h[0:8]))
	wantCRC := binary.LittleEndian.Uint32(h[8:12])

	pbuf := make([]byte, storedLen)
	if _, err := l.ReadAt(pbuf, off); err != nil {
		t.Fatalf("ReadAt payload: %v", err)
	}
	if crc32.ChecksumIEEE(pbuf) != wantCRC {
		t.Fatalf("crc mismatch")
	}
	if !bytes.Equal(pbuf, msg) {
		t.Fatalf("payload mismatch via raw read")
	}
}

func TestLargePayloads_ManyRecords(t *testing.T) {
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "snappy", retain: true, checkptK: 8})
	defer l.Close()

	N := 200
	keep := make([][]byte, 0, N)
	for i := 0; i < N; i++ {
		keep = append(keep, payload(1024+((i%5)*137)))
	}

	first, last, err := l.AppendBatch(keep...)
	if err != nil {
		t.Fatalf("AppendBatch: %v", err)
	}
	if first != 1 || last != uint64(N) {
		t.Fatalf("range got [%d,%d] want [1,%d]", first, last, N)
	}

	// sample a few positions
	for _, i := range []int{1, 2, 7, 8, 9, 63, 64, 127, 128, 199, 200} {
		out, err := l.Read(uint64(i))
		if err != nil {
			t.Fatalf("Read(%d): %v", i, err)
		}
		if !bytes.Equal(out, keep[i-1]) {
			t.Fatalf("rec %d mismatch", i)
		}
	}
}

func TestDeleteOnClose(t *testing.T) {
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "none", deleteOnClose: true})
	if _, err := l.Append([]byte("bye")); err != nil {
		t.Fatalf("append: %v", err)
	}
	if err := l.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if _, err := os.Stat(p); !os.IsNotExist(err) {
		t.Fatalf("expected file to be removed, stat err=%v", err)
	}
}

func TestConcurrencySmoke(t *testing.T) {
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "snappy", retain: true, checkptK: 1, noSync: true, bufSize: 1 << 20})
	defer l.Close()

	// A light race-ish smoke test to ensure no panics under mixed ops.
	done := make(chan struct{})
	go func() {
		for i := 0; i < 200; i++ {
			_, _ = l.Append(payload(300))
			time.Sleep(time.Millisecond)
		}
		close(done)
	}()

	readErrs := 0
	for i := 0; i < 200; i++ {
		idx := l.LastIndex()
		if idx > 0 {
			if _, err := l.Read(idx); err != nil && !errors.Is(err, io.EOF) {
				readErrs++
			}
		}
		time.Sleep(time.Millisecond)
	}
	<-done
	// non-fatal; goal is not to race test but to catch obvious API panics
	if readErrs > 0 {
		t.Logf("non-fatal read errors observed during concurrent appends: %d", readErrs)
	}
}
