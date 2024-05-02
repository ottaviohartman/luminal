use criterion::{criterion_group, criterion_main, Criterion};
use luminal::prelude::*;

fn bench_matmul() {
	let mut cx = Graph::new();
	let a = cx.tensor::<(Dyn<'M'>, Dyn<'K'>)>();
	let b = cx.tensor::<(Dyn<'K'>, Dyn<'N'>)>();
	let mut c = a.matmul(b).retrieve();

	cx.compile(CPUCompiler::default(), &mut c);

	let d_dev = dfdx::prelude::Cpu::default();
	for m in (1..23).step_by(4) {
		for k in (1..35).step_by(3) {
			for n in (1..70).step_by(7) {
				let mut rng = StdRng::seed_from_u64(0);
				let a_data = random_vec_rng(m * k, &mut rng);
				let b_data = random_vec_rng(k * n, &mut rng);
				a.set_dyn(a_data.clone(), &[m, k]);
				b.set_dyn(b_data.clone(), &[k, n]);

				cx.execute();

				let d_a = d_dev.tensor_from_vec(a_data, (m, k));
				let d_b = d_dev.tensor_from_vec(b_data, (k, n));
				let d_c = d_a.matmul(d_b);

				assert_close_precision(&c.data(), &d_c.to_dtype::<f32>().as_vec(), 1e-2);
				c.drop();
			}
		}
	}
}

fn criterion_benchmark(c: &mut Criterion) {
	let a = [[1.0], [2.0], [3.0]];
	let b = [[1.0, 2.0, 3.0, 4.0]];
    c.bench_function("new graph", |b| b.iter(|| bench()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
