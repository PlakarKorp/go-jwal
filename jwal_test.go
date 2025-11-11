package jwal

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func tmp(t *testing.T) string {
	t.Helper()
	f, err := os.CreateTemp("", "jwal-*")
	if err != nil {
		t.Fatal(err)
	}
	name := f.Name()
	_ = f.Close()
	_ = os.Remove(name) // we’ll recreate
	return name
}

func fill(b []byte, v byte) {
	for i := range b {
		b[i] = v
	}
}

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

func TestRoundTrip_AllModes(t *testing.T) {
	codecs := []string{"none", "snappy"}
	modes := []struct {
		retain bool
		k      uint
	}{
		{false, 0}, {true, 1}, {true, 64},
	}

	for _, c := range codecs {
		for _, m := range modes {
			t.Run(fmt.Sprintf("codec=%s retain=%v k=%d", c, m.retain, m.k), func(t *testing.T) {
				p := tmp(t)
				l := mustOpen(t, p, openOpt{comp: c, retain: m.retain, checkptK: m.k})

				// append singles
				recs := [][]byte{}
				for i := 0; i < 257; i++ {
					buf := make([]byte, 123+i%5)
					fill(buf, byte(i))
					recs = append(recs, append([]byte(nil), buf...))
					if _, _, err := l.AppendBatch(buf); err != nil {
						t.Fatal(err)
					}
				}
				// append batched
				if _, _, err := l.AppendBatch(recs...); err != nil {
					t.Fatal(err)
				}

				// reopen (persistence)
				_ = l.Close()
				l = mustOpen(t, p, openOpt{comp: "none", retain: true, checkptK: 1}) // codec ignored on existing
				last := l.LastIndex()
				want := uint64(257 + len(recs))
				if last != want {
					t.Fatalf("LastIndex=%d want %d", last, want)
				}

				// verify
				for i := uint64(1); i <= last; i++ {
					got, err := l.Read(i)
					if err != nil {
						t.Fatalf("Read(%d): %v", i, err)
					}
					// first 257 were various lengths; next chunk equals 'recs[i-258]'
					// recompute expected:
					var exp []byte
					if i <= 257 {
						n := 123 + int((i-1)%5)
						exp = make([]byte, n)
						fill(exp, byte(i-1))
					} else {
						exp = recs[i-258]
					}
					if !bytes.Equal(got, exp) {
						t.Fatalf("mismatch at %d", i)
					}
				}
			})
		}
	}
}

func TestRecover_TornTail_And_CRC(t *testing.T) {
	p := tmp(t)
	l := mustOpen(t, p, openOpt{comp: "snappy", retain: false})
	// write some good records
	for i := 0; i < 1000; i++ {
		buf := make([]byte, 512)
		fill(buf, byte(i))
		if _, _, err := l.AppendBatch(buf); err != nil {
			t.Fatal(err)
		}
	}
	_ = l.Sync()
	_ = l.Close()

	// append a partial (torn) record directly to file
	f, err := os.OpenFile(p, os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	st, _ := f.Stat()
	_ = st
	// write only a header without full payload
	hdr := make([]byte, recordHdrSize)
	binary.LittleEndian.PutUint64(hdr[0:8], 4096) // claims payload len
	_, _ = f.Seek(0, io.SeekEnd)
	_, _ = f.Write(hdr[:10]) // partial header
	_ = f.Close()

	// reopen must truncate to last good
	l = mustOpen(t, p, openOpt{})
	if last := l.LastIndex(); last != 1000 {
		t.Fatalf("recover LastIndex=%d want 1000", last)
	}

	// now flip a byte in the middle payload and ensure recovery truncates there
	_ = l.Close()
	f, _ = os.OpenFile(p, os.O_RDWR, 0o600)
	// locate record ~500 by scanning quickly
	// simple: corrupt just after dataBase + 500*(hdr+payloadApprox)
	// (robust way: use Iter)
	_ = f.Close()
	// Better: use Iter to find payload offset and flip one byte:
	l = mustOpen(t, p, openOpt{retain: true, checkptK: 1})
	off, err := l.DataOffset(700)
	if err != nil {
		t.Fatal(err)
	}
	rf, _ := os.OpenFile(p, os.O_RDWR, 0o600)
	b := []byte{0xFF}
	if _, err := rf.WriteAt(b, off); err != nil {
		t.Fatal(err)
	}
	_ = rf.Close()
	_ = l.Close()

	l = mustOpen(t, p, openOpt{})
	if last := l.LastIndex(); last != 699 {
		t.Fatalf("after CRC corruption LastIndex=%d want 699", last)
	}
}

func TestParallel_Writes_And_Reads(t *testing.T) {
	p := tmp(t)
	l := mustOpen(t, p, openOpt{comp: "snappy", retain: true, checkptK: 64, noSync: true})
	defer l.Close()

	const writers = 8
	const perWriter = 50_000
	const recSize = 256
	var wg sync.WaitGroup
	wg.Add(writers)

	for w := 0; w < writers; w++ {
		go func() {
			defer wg.Done()
			// preallocate batch
			batch := make([][]byte, 256)
			zero := make([]byte, recSize)
			for i := range batch {
				batch[i] = zero
			}
			written := 0
			for written < perWriter {
				n := 256
				if perWriter-written < n {
					n = perWriter - written
				}
				if _, _, err := l.AppendBatch(batch[:n]...); err != nil {
					t.Error(err)
					return
				}
				written += n
			}
		}()
	}
	wg.Wait()
	_ = l.Sync()

	last := l.LastIndex()
	if last != writers*perWriter {
		t.Fatalf("LastIndex=%d want %d", last, writers*perWriter)
	}

	// parallel verify reads
	l2 := mustOpen(t, p, openOpt{retain: true, checkptK: 1})
	var vr sync.WaitGroup
	const readers = 16
	vr.Add(readers)
	for r := 0; r < readers; r++ {
		go func(r int) {
			defer vr.Done()
			dst := make([]byte, 0, recSize)
			for i := uint64(r + 1); i <= uint64(last); i += readers {
				out, err := l2.ReadInto(i, dst[:0])
				if err != nil {
					t.Error(err)
					return
				}
				for _, bb := range out {
					if bb != 0 {
						t.Fatalf("non-zero at %d", i)
					}
				}
			}
		}(r)
	}
	vr.Wait()
}

func TestTruncate_Then_Append_Random(t *testing.T) {
	p := tmp(t)
	l := mustOpen(t, p, openOpt{retain: true, checkptK: 1})
	defer l.Close()

	rnd := rand.New(rand.NewSource(1))
	var logical [][]byte

	for step := 0; step < 200; step++ {
		switch rnd.Intn(3) {
		case 0: // append a random batch
			k := 1 + rnd.Intn(50)
			batch := make([][]byte, k)
			for i := range batch {
				n := 1 + rnd.Intn(1024)
				b := make([]byte, n)
				rnd.Read(b)
				logical = append(logical, b)
				batch[i] = b
			}
			if _, _, err := l.AppendBatch(batch...); err != nil {
				t.Fatal(err)
			}
		case 1: // truncate back to random index
			if len(logical) == 0 {
				continue
			}
			keep := rnd.Intn(len(logical) + 1)
			if err := l.TruncateBack(uint64(keep)); err != nil {
				t.Fatal(err)
			}
			logical = logical[:keep]
		default:
			_ = l.Sync()
		}
	}

	_ = l.Sync()
	_ = l.Close()
	l = mustOpen(t, p, openOpt{retain: true, checkptK: 1})
	if int(l.LastIndex()) != len(logical) {
		t.Fatalf("LastIndex=%d want %d", l.LastIndex(), len(logical))
	}

	for i := 1; i <= len(logical); i++ {
		got, err := l.Read(uint64(i))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, logical[i-1]) {
			t.Fatalf("mismatch at %d", i)
		}
	}
}

func FuzzAppendRead(f *testing.F) {
	f.Add([]byte("a"), []byte("b"))
	f.Add([]byte{0}, make([]byte, 1024))
	f.Fuzz(func(t *testing.T, a, b []byte) {
		p := tmp(t)
		l := mustOpen(t, p, openOpt{comp: "snappy", retain: true, checkptK: 1})
		defer l.Close()
		if _, _, err := l.AppendBatch(a, b); err != nil {
			t.Skip()
		}
		_ = l.Sync()
		for i := 1; i <= 2; i++ {
			got, err := l.Read(uint64(i))
			if err != nil {
				t.Fatal(err)
			}
			exp := [][]byte{a, b}[i-1]
			if !bytes.Equal(got, exp) {
				t.Fatalf("bad roundtrip")
			}
		}
	})
}

func TestAppendBatch_EmptyAndWriteBatch_Empty(t *testing.T) {
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "snappy", retain: true, checkptK: 1})
	defer l.Close()

	if got := l.LastIndex(); got != 0 {
		t.Fatalf("LastIndex=%d want 0", got)
	}

	// AppendBatch with no args should be a no-op but return current indices.
	first, last, err := l.AppendBatch()
	if err != nil {
		t.Fatalf("AppendBatch(empty): %v", err)
	}
	if first != 0 || last != 0 || l.LastIndex() != 0 {
		t.Fatalf("empty AppendBatch changed state: first=%d last=%d lastIdx=%d", first, last, l.LastIndex())
	}

	// WriteBatch with nil/empty should also be a no-op.
	var b Batch
	first, last, err = l.WriteBatch(&b)
	if err != nil {
		t.Fatalf("WriteBatch(empty): %v", err)
	}
	if first != 0 || last != 0 || l.LastIndex() != 0 {
		t.Fatalf("empty WriteBatch changed state: first=%d last=%d lastIdx=%d", first, last, l.LastIndex())
	}
}

func TestBatch_API_Add_Reset_WriteBatch(t *testing.T) {
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "none", retain: true, checkptK: 1})
	defer l.Close()

	var b Batch
	b.Add([]byte("a"))
	b.Add([]byte("bb"))
	if b.Len() != 2 {
		t.Fatalf("Len=%d want 2", b.Len())
	}
	first, last, err := l.WriteBatch(&b)
	if err != nil {
		t.Fatalf("WriteBatch: %v", err)
	}
	if first != 1 || last != 2 {
		t.Fatalf("range got [%d,%d] want [1,2]", first, last)
	}

	// Reset and reuse
	b.Reset()
	if b.Len() != 0 {
		t.Fatalf("Reset didn't clear")
	}
	b.Add([]byte("ccc"))
	b.Add([]byte("dddd"))
	first, last, err = l.WriteBatch(&b)
	if err != nil {
		t.Fatalf("WriteBatch#2: %v", err)
	}
	if first != 3 || last != 4 {
		t.Fatalf("range2 got [%d,%d] want [3,4]", first, last)
	}

	// Verify all 4
	exp := [][]byte{[]byte("a"), []byte("bb"), []byte("ccc"), []byte("dddd")}
	for i := 1; i <= 4; i++ {
		got, err := l.Read(uint64(i))
		if err != nil {
			t.Fatalf("Read(%d): %v", i, err)
		}
		if !bytes.Equal(got, exp[i-1]) {
			t.Fatalf("rec %d mismatch", i)
		}
	}
}

func TestReadInto_ReusesDstWithSnappy(t *testing.T) {
	// Exercise snappy path with ulen>0; ensure content and length are correct.
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "snappy", retain: true, checkptK: 1})
	defer l.Close()

	orig := make([]byte, 64<<10) // 64 KiB
	for i := range orig {
		orig[i] = byte(i)
	}
	if _, err := l.Append(orig); err != nil {
		t.Fatalf("append: %v", err)
	}

	// Provide a dst with enough capacity to *allow* reuse (not asserted).
	dst := make([]byte, 0, len(orig)+128)

	out, err := l.ReadInto(1, dst[:0])
	if err != nil {
		t.Fatalf("ReadInto: %v", err)
	}
	if len(out) != len(orig) {
		t.Fatalf("len(out)=%d want %d", len(out), len(orig))
	}
	if !bytes.Equal(out, orig) {
		t.Fatalf("payload mismatch")
	}
}

func TestIter_FromMiddleIndex(t *testing.T) {
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "none", retain: true, checkptK: 1})
	defer l.Close()

	for i := 0; i < 10; i++ {
		if _, err := l.Append([]byte{byte(i)}); err != nil {
			t.Fatalf("append: %v", err)
		}
	}

	it, err := l.Iter(5) // start from middle
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
		if idx < 5 || idx > 10 {
			t.Fatalf("idx out of expected range: %d", idx)
		}
		got = append(got, b)
	}
	if len(got) != 6 {
		t.Fatalf("iter len=%d want 6", len(got))
	}
}

func TestDataOffset_NoIndex_LinearScan(t *testing.T) {
	// Verify locateDataOffsetLocked path when retainIndex=false (linear header walks).
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "none", retain: false})
	defer l.Close()

	// Variable sizes to ensure header-walk math is exercised.
	var recs [][]byte
	for i := 0; i < 7; i++ {
		rec := bytes.Repeat([]byte{byte(0xA0 + i)}, 50+i*13)
		recs = append(recs, rec)
		if _, err := l.Append(rec); err != nil {
			t.Fatalf("append: %v", err)
		}
	}

	for i := 1; i <= len(recs); i++ {
		off, err := l.DataOffset(uint64(i))
		if err != nil {
			t.Fatalf("DataOffset(%d): %v", i, err)
		}

		// Read back via offsets and validate CRC and bytes.
		h := make([]byte, recordHdrSize)
		if _, err := l.ReadAt(h, off-recordHdrSize); err != nil {
			t.Fatalf("ReadAt hdr: %v", err)
		}
		storedLen := int(binary.LittleEndian.Uint64(h[0:8]))
		wantCRC := binary.LittleEndian.Uint32(h[8:12])

		pbuf := make([]byte, storedLen)
		if _, err := l.ReadAt(pbuf, off); err != nil {
			t.Fatalf("ReadAt payload: %v", err)
		}
		if crc32.ChecksumIEEE(pbuf) != wantCRC {
			t.Fatalf("crc mismatch at %d", i)
		}
		if !bytes.Equal(pbuf, recs[i-1]) {
			t.Fatalf("payload mismatch at %d", i)
		}
	}
}

func TestCompressionHeaderHonoredOnReopen(t *testing.T) {
	// Write with snappy, close, reopen with opts.Compression="none", then append and read everything.
	// Works only if Open picks compCodec from file header (your fix).
	p := tmpPath(t)

	// phase 1: write with snappy
	l := mustOpen(t, p, openOpt{comp: "snappy"})
	if _, err := l.Append([]byte("s1")); err != nil {
		t.Fatalf("append: %v", err)
	}
	if err := l.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	// phase 2: reopen with "none" in options (should be ignored) and append again
	l = mustOpen(t, p, openOpt{comp: "none"})
	defer l.Close()
	if _, err := l.Append([]byte("s2")); err != nil {
		t.Fatalf("append2: %v", err)
	}

	// read both
	got1, err := l.Read(1)
	if err != nil {
		t.Fatalf("Read1: %v", err)
	}
	got2, err := l.Read(2)
	if err != nil {
		t.Fatalf("Read2: %v", err)
	}
	if string(got1) != "s1" || string(got2) != "s2" {
		t.Fatalf("reopen or codec mismatch: %q %q", got1, got2)
	}
}

func TestOpenWithShortHeader(t *testing.T) {
	// Create a file smaller than walHdrSize and ensure Open treats as new and writes header.
	p := tmpPath(t)
	f, err := os.Create(p)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	// write fewer than walHdrSize bytes
	if _, err := f.Write(make([]byte, walHdrSize/2)); err != nil {
		t.Fatalf("seed: %v", err)
	}
	_ = f.Close()

	l := mustOpen(t, p, openOpt{comp: "none"})
	defer l.Close()

	if l.LastIndex() != 0 {
		t.Fatalf("LastIndex=%d want 0 for short-header init", l.LastIndex())
	}
	if got := fileSize(t, p); got < walHdrSize {
		t.Fatalf("file not extended to header, size=%d", got)
	}
}

func TestReadIndexZero_And_Beyond(t *testing.T) {
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "none"})
	defer l.Close()
	if _, err := l.Append([]byte("x")); err != nil {
		t.Fatalf("append: %v", err)
	}

	if _, err := l.Read(0); !errors.Is(err, io.EOF) {
		t.Fatalf("Read(0) should EOF, got %v", err)
	}
	if _, err := l.Read(2); !errors.Is(err, io.EOF) {
		t.Fatalf("Read(2) should EOF, got %v", err)
	}
}

func TestOpen_InvalidMagic(t *testing.T) {
	p := tmpPath(t)

	// Write a bogus 16-byte header with wrong magic.
	f, err := os.Create(p)
	if err != nil {
		t.Fatal(err)
	}
	var hdr [16]byte
	binary.LittleEndian.PutUint32(hdr[0:4], 0xDEADBEEF) // bad magic
	binary.LittleEndian.PutUint16(hdr[4:6], version)
	hdr[6] = crcIEEE
	hdr[7] = compNone
	if _, err := f.WriteAt(hdr[:], 0); err != nil {
		t.Fatal(err)
	}
	_ = f.Close()

	if _, err := Open(p, &Options{}); err == nil {
		t.Fatalf("expected error for invalid magic")
	}
}

func TestOpen_UnsupportedVersion(t *testing.T) {
	p := tmpPath(t)

	f, err := os.Create(p)
	if err != nil {
		t.Fatal(err)
	}
	var hdr [16]byte
	binary.LittleEndian.PutUint32(hdr[0:4], magicJWAL)
	binary.LittleEndian.PutUint16(hdr[4:6], version+1) // unsupported
	hdr[6] = crcIEEE
	hdr[7] = compNone
	if _, err := f.WriteAt(hdr[:], 0); err != nil {
		t.Fatal(err)
	}
	_ = f.Close()

	if _, err := Open(p, &Options{}); err == nil {
		t.Fatalf("expected error for unsupported version")
	}
}

func TestRead_CRCMismatch_ReturnsUnexpectedEOF(t *testing.T) {
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "snappy", retain: true, checkptK: 1})
	defer l.Close()

	data := payload(8 << 10)
	if _, err := l.Append(data); err != nil {
		t.Fatal(err)
	}
	_ = l.Sync()

	// Find payload offset and flip a byte after open.
	off, err := l.DataOffset(1)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(p, os.O_RDWR, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var b [1]byte
	if _, err := f.ReadAt(b[:], off); err != nil {
		t.Fatal(err)
	}
	b[0] ^= 0xFF
	if _, err := f.WriteAt(b[:], off); err != nil {
		t.Fatal(err)
	}

	// Now a Read should fail CRC.
	if _, err := l.Read(1); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("want ErrUnexpectedEOF, got %v", err)
	}
	// ReadInto too.
	if _, err := l.ReadInto(1, nil); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("ReadInto: want ErrUnexpectedEOF, got %v", err)
	}
}

func TestIterator_NextInto_EndBoundary(t *testing.T) {
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "none", retain: true, checkptK: 1})
	defer l.Close()

	in := [][]byte{[]byte("x"), []byte("yy")}
	if _, _, err := l.AppendBatch(in...); err != nil {
		t.Fatal(err)
	}

	it, err := l.Iter(1)
	if err != nil {
		t.Fatal(err)
	}

	dst := make([]byte, 0, 8)
	out, idx, err := it.NextInto(dst[:0])
	if err != nil || idx != 1 || !bytes.Equal(out, in[0]) {
		t.Fatalf("first: idx=%d err=%v ok=%v", idx, err, bytes.Equal(out, in[0]))
	}
	out, idx, err = it.NextInto(dst[:0])
	if err != nil || idx != 2 || !bytes.Equal(out, in[1]) {
		t.Fatalf("second: idx=%d err=%v ok=%v", idx, err, bytes.Equal(out, in[1]))
	}
	// Now we should hit EOF.
	if _, _, err := it.NextInto(dst[:0]); !errors.Is(err, io.EOF) {
		t.Fatalf("want EOF at end, got %v", err)
	}
}

func TestDataOffset_OutOfRange(t *testing.T) {
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "none"})
	defer l.Close()

	if _, err := l.DataOffset(0); err == nil {
		t.Fatalf("DataOffset(0) should error")
	}
	if _, err := l.Append([]byte("a")); err != nil {
		t.Fatal(err)
	}
	if _, err := l.DataOffset(2); err == nil {
		t.Fatalf("DataOffset(2) should error")
	}
}

func TestTruncateBack_SparseIndexTrim(t *testing.T) {
	p := tmpPath(t)
	l := mustOpen(t, p, openOpt{comp: "snappy", retain: true, checkptK: 4})
	defer l.Close()

	// 10 records total -> checkpoints at 1 and 5 and 9 (1+4+4)
	var recs [][]byte
	for i := 0; i < 10; i++ {
		recs = append(recs, []byte{byte(i)})
	}
	if _, _, err := l.AppendBatch(recs...); err != nil {
		t.Fatal(err)
	}

	// Truncate to 7; last checkpoint should still exist (at 5), next (at 9) removed.
	if err := l.TruncateBack(7); err != nil {
		t.Fatal(err)
	}
	if l.LastIndex() != 7 {
		t.Fatalf("LastIndex=%d want 7", l.LastIndex())
	}

	// Reopen and read around checkpoint boundaries to ensure locate still works.
	_ = l.Close()
	l = mustOpen(t, p, openOpt{comp: "snappy", retain: true, checkptK: 4})
	defer l.Close()

	for _, i := range []int{1, 4, 5, 6, 7} {
		got, err := l.Read(uint64(i))
		if err != nil {
			t.Fatalf("Read(%d): %v", i, err)
		}
		if len(got) != 1 || got[0] != byte(i-1) {
			t.Fatalf("rec %d mismatch", i)
		}
	}
}

// --- PARALLEL READS ----------------------------------------------------------

func prefillForReads(b *testing.B, path string, recSize, recs int, comp string, indexK uint) {
	l, err := Open(path, &Options{
		NoSync:             true,
		BufferSize:         256 << 10,
		RetainIndex:        true,
		CheckpointInterval: indexK, // often 1 for O(1) reads
		Compression:        comp,
	})
	if err != nil {
		b.Fatalf("Open(prefill): %v", err)
	}
	defer l.Close()

	blk := payloadOf(recSize)
	// fill in big batches to speed up prefill
	batch := make([][]byte, 256)
	for i := range batch {
		batch[i] = blk
	}
	written := 0
	for written < recs {
		n := len(batch)
		if recs-written < n {
			n = recs - written
		}
		if _, _, err := l.AppendBatch(batch[:n]...); err != nil {
			b.Fatalf("AppendBatch(prefill): %v", err)
		}
		written += n
	}
	_ = l.Sync()
}

// Full-index (k=1) + ReadInto (zero-alloc hot path)
func BenchmarkReadParallel_ReadInto_FullIndex(b *testing.B) {
	const recSize = 1024
	const prefill = 1 << 16 // 65,536 records

	path := benchPath(b)
	prefillForReads(b, path, recSize, prefill, "snappy", 1)

	l, err := Open(path, &Options{
		NoSync:             true,
		BufferSize:         256 << 10,
		RetainIndex:        true,
		CheckpointInterval: 1, // full index => O(1)
	})
	if err != nil {
		b.Fatalf("Open(read): %v", err)
	}
	defer l.Close()

	last := l.LastIndex()
	if last == 0 {
		b.Fatal("empty WAL after prefill")
	}

	var counter atomic.Uint64
	b.SetBytes(int64(recSize))
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		dst := make([]byte, 0, recSize)
		for pb.Next() {
			i := (counter.Add(1)-1)%last + 1
			out, err := l.ReadInto(i, dst[:0])
			if err != nil {
				b.Fatalf("ReadInto(%d): %v", i, err)
			}
			if len(out) != recSize {
				b.Fatalf("size=%d want %d", len(out), recSize)
			}
		}
	})
	b.StopTimer()
}

// Full-index (k=1) + Read (allocating)
func BenchmarkReadParallel_Read_FullIndex(b *testing.B) {
	const recSize = 1024
	const prefill = 1 << 16

	path := benchPath(b)
	prefillForReads(b, path, recSize, prefill, "snappy", 1)

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

	last := l.LastIndex()
	if last == 0 {
		b.Fatal("empty WAL after prefill")
	}

	var counter atomic.Uint64
	b.SetBytes(int64(recSize))
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			i := (counter.Add(1)-1)%last + 1
			buf, err := l.Read(i)
			if err != nil {
				b.Fatalf("Read(%d): %v", i, err)
			}
			if len(buf) != recSize {
				b.Fatalf("size=%d want %d", len(buf), recSize)
			}
		}
	})
	b.StopTimer()
}

// Random-access indices (full index) using ReadInto
func BenchmarkReadParallel_Random_ReadInto(b *testing.B) {
	const recSize = 2048
	const prefill = 1 << 15

	path := benchPath(b)
	prefillForReads(b, path, recSize, prefill, "snappy", 1)

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

	last := int(l.LastIndex())
	if last == 0 {
		b.Fatal("empty WAL after prefill")
	}

	b.SetBytes(int64(recSize))
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		dst := make([]byte, 0, recSize)
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		for pb.Next() {
			i := 1 + r.Intn(last)
			out, err := l.ReadInto(uint64(i), dst[:0])
			if err != nil {
				b.Fatalf("ReadInto(%d): %v", i, err)
			}
			if len(out) != recSize {
				b.Fatalf("size=%d want %d", len(out), recSize)
			}
		}
	})
	b.StopTimer()
}

// Sparse index (k=4096) to expose scan-ahead cost under parallel reads.
func BenchmarkReadParallel_SparseIndex_ReadInto(b *testing.B) {
	const recSize = 512
	const prefill = 1 << 15

	path := benchPath(b)
	// Prefill with sparse checkpoints (also used for read)
	prefillForReads(b, path, recSize, prefill, "snappy", 4096)

	l, err := Open(path, &Options{
		NoSync:             true,
		BufferSize:         256 << 10,
		RetainIndex:        true,
		CheckpointInterval: 4096, // scan from nearest checkpoint
	})
	if err != nil {
		b.Fatalf("Open(read): %v", err)
	}
	defer l.Close()

	last := l.LastIndex()
	if last == 0 {
		b.Fatal("empty WAL after prefill")
	}

	var counter atomic.Uint64
	b.SetBytes(int64(recSize))
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		dst := make([]byte, 0, recSize)
		for pb.Next() {
			i := (counter.Add(1)-1)%last + 1
			out, err := l.ReadInto(i, dst[:0])
			if err != nil {
				b.Fatalf("ReadInto(%d): %v", i, err)
			}
			if len(out) != recSize {
				b.Fatalf("size=%d want %d", len(out), recSize)
			}
		}
	})
	b.StopTimer()
}
