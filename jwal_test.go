package jwal

import (
	"crypto/rand"
	"io"
	"path/filepath"
	"testing"
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
