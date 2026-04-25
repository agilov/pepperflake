use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pepperflake::{generate, is_valid, timestamp_millis};

fn bench_generate(c: &mut Criterion) {
    // We only benchmark the generation loop.
    // black_box ensures the compiler doesn't optimize away the function call.
    c.bench_function("generate", |b| b.iter(|| black_box(generate())));
}

fn bench_is_valid(c: &mut Criterion) {
    // Generate a valid ID once *outside* the benchmark loop so we are
    // strictly measuring the validation math, not the generation.
    let id = generate();

    c.bench_function("is_valid", |b| {
        b.iter(|| black_box(is_valid(black_box(id))))
    });
}

fn bench_timestamp_millis(c: &mut Criterion) {
    let id = generate();

    c.bench_function("timestamp_millis", |b| {
        b.iter(|| black_box(timestamp_millis(black_box(id))))
    });
}

// Group the benchmarks and register the main entry point
criterion_group!(
    benches,
    bench_generate,
    bench_is_valid,
    bench_timestamp_millis
);
criterion_main!(benches);
