use crate::cuda_bindings::*;
use crate::torch_call::{TensorInfo, TorchCall, TorchCallInfo};
use crate::{assert_cuda, estimate};
use lru::LruCache;
use std::collections::{BTreeMap, HashMap};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::num::NonZeroUsize;
use std::ptr;
use std::time::Duration;
use tch::{self, Device, Kind, Tensor};

enum KindRange {
    Integer,
    Bool,
    Float,
}

fn kind_range(kind: Kind) -> KindRange {
    match kind {
        Kind::Bool => KindRange::Bool,
        Kind::Uint8
        | Kind::Int8
        | Kind::Int16
        | Kind::UInt16
        | Kind::Int
        | Kind::Int64
        | Kind::UInt32
        | Kind::UInt64 => KindRange::Integer,
        _ => KindRange::Float,
    }
}

fn tensor_cache_key(info: &TensorInfo) -> Option<(i64, Kind)> {
    match kind_range(info.dtype) {
        KindRange::Float => Some((info.shape.iter().product(), info.dtype)),
        // Integer and boolean contents can determine indices, masks, reduction
        // sizes, and control flow. Reusing them by only (numel, dtype) aliases
        // unrelated semantics, so value-sensitive tensors are always fresh.
        KindRange::Integer | KindRange::Bool => None,
    }
}

fn allocate_class_index_target(
    input_info: &TensorInfo,
    target_info: &TensorInfo,
    device: Device,
) -> Tensor {
    let class_dimension = if input_info.shape.len() == 1 { 0 } else { 1 };
    let n_classes = *input_info
        .shape
        .get(class_dimension)
        .expect("cross-entropy input must have a class dimension");
    assert!(
        n_classes > 0,
        "cross-entropy class dimension must be positive"
    );

    // Class-index targets affect the loss value, not the measured kernel shape.
    // Allocate them independently so unrelated integer tensors with the same
    // (numel, dtype) cache key can never become invalid class indices.
    Tensor::zeros(target_info.shape.as_slice(), (target_info.dtype, device))
}

/// Measure a torch op with `n` warmup + 1 measured iteration.
/// Skips `tch::autocast` when inputs are already bf16/fp16 — autocast
/// adds ~6× dispatch overhead for tensors that won't be dtype-converted,
/// inflating both cudaEvent and CUPTI measurements. Only applies autocast
/// for fp32 inputs where mixed-precision is actually useful.
macro_rules! estimate_torch {
    ($n:expr, $e:expr, $dtype:expr) => {{
        if $dtype == tch::Kind::Float || $dtype == tch::Kind::Double {
            tch::autocast(true, || estimate!($n, $e))
        } else {
            estimate!($n, $e)
        }
    }};
    // Backward-compatible: no dtype → no autocast (default inference path).
    ($n:expr, $e:expr) => {{
        estimate!($n, $e)
    }};
}

pub struct TorchEstimator {
    tensor_cache: LruCache<(i64, Kind), Tensor>,
    compute_cache: HashMap<TorchCallInfo, Duration>,
    sequence_cache: BTreeMap<u64, Vec<Duration>>,
}

impl TorchEstimator {
    pub fn new() -> Self {
        let mat = Tensor::randn(&[1024, 1024], (Kind::Float, Device::Cuda(0)));
        let _ = mat.mm(&mat); // initialization

        Self {
            tensor_cache: LruCache::new(NonZeroUsize::new(32).unwrap()),
            compute_cache: HashMap::new(),
            sequence_cache: BTreeMap::new(),
        }
    }

    fn allocate_on(&mut self, info: &TensorInfo, device: Device) -> Tensor {
        let shape = info.shape.as_slice();
        let kind = info.dtype;
        let cache_key = tensor_cache_key(info);
        if let Some(key) = cache_key {
            if let Some(t) = self.tensor_cache.get(&key) {
                return t.contiguous().view(shape);
            }
        }

        let t = match kind_range(kind) {
            KindRange::Bool => Tensor::randint(2, shape, (kind, device)),
            KindRange::Integer => Tensor::randint(128, shape, (kind, device)),
            KindRange::Float => Tensor::randn(shape, (kind, device)),
        };
        if let Some(key) = cache_key {
            self.tensor_cache.put(key, t.contiguous());
        }
        t
    }

    fn allocate(&mut self, info: &TensorInfo) -> Tensor {
        self.allocate_on(info, Device::Cuda(0))
    }

    fn allocate_list(&mut self, info: &[TensorInfo]) -> Vec<Tensor> {
        info.iter().map(|info| self.allocate(info)).collect()
    }

    fn cache(&mut self, t: Tensor) {
        if let Device::Cuda(0) = t.device() {
            let kind = t.kind();
            let info = TensorInfo {
                shape: t.size(),
                dtype: kind,
            };
            if let Some(key) = tensor_cache_key(&info) {
                self.tensor_cache.put(key, t.contiguous());
            }
        }
    }

    fn run(&mut self, niter: i32, call: &TorchCallInfo) -> Duration {
        match call {
            TorchCallInfo::MM(info1, info2) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(niter, t1.mm(&t2));
                self.cache(result);
                dur
            }
            TorchCallInfo::MatMul(info1, info2) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(niter, t1.matmul(&t2));
                self.cache(result);
                dur
            }
            TorchCallInfo::Linear(info1, info2, bias_info) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let bias = bias_info.as_ref().map(|info| self.allocate(info));
                let (result, dur) =
                    estimate_torch!(niter, t1.linear::<&Tensor>(&t2, bias.as_ref()), info2.dtype);
                self.cache(result);
                dur
            }
            TorchCallInfo::BMM(info1, info2) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(niter, t1.bmm(&t2));
                self.cache(result);
                dur
            }
            TorchCallInfo::AddMM(info1, info2, info3) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let t3 = self.allocate(info3);
                let (result, dur) = estimate_torch!(niter, t1.addmm(&t2, &t3));
                self.cache(result);
                dur
            }
            TorchCallInfo::BAddBMM(info1, info2, info3) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let t3 = self.allocate(info3);
                let (result, dur) = estimate_torch!(niter, t1.baddbmm(&t2, &t3, 1, 1));
                self.cache(result);
                dur
            }
            TorchCallInfo::Mul(info1, info2) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(niter, t1.g_mul(&t2));
                self.cache(result);
                dur
            }
            TorchCallInfo::MulScalar(info) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.multiply_scalar(2));
                self.cache(result);
                dur
            }
            TorchCallInfo::Mul_(info1, info2) => {
                let mut t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(niter, t1.g_mul_(&t2));
                self.cache(result);
                dur
            }
            TorchCallInfo::MulScalar_(info) => {
                let mut t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.multiply_scalar_(2));
                self.cache(result);
                dur
            }
            TorchCallInfo::ForeachMul_(info_list1, info_list2) => {
                let t_list2 = self.allocate_list(info_list2);
                let t_list1 = self.allocate_list(info_list1);
                let (_, dur) = estimate_torch!(
                    niter,
                    Tensor::f_internal_foreach_mul_list_(&t_list1, &t_list2)
                );
                dur
            }
            TorchCallInfo::ForeachMulScalar_(info_list) => {
                let t_list = self.allocate_list(info_list);
                let (_, dur) = estimate_torch!(niter, Tensor::f_internal_foreach_mul_(&t_list, 2));
                dur
            }
            TorchCallInfo::Add(info1, info2) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(niter, t1.g_add(&t2));
                self.cache(result);
                dur
            }
            TorchCallInfo::Add_(info1, info2) => {
                let mut t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(niter, t1.g_add_(&t2));
                self.cache(result);
                dur
            }
            TorchCallInfo::Div(info1, info2) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(niter, t1.g_div(&t2));
                self.cache(result);
                dur
            }
            TorchCallInfo::DivScalar(info) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.divide_scalar(2));
                self.cache(result);
                dur
            }
            TorchCallInfo::Pow(info) => {
                let mut t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.pow_(2));
                self.cache(result);
                dur
            }
            TorchCallInfo::AddCMul_(info1, info2, info3) => {
                let mut t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let t3 = self.allocate(info3);
                let (result, dur) = estimate_torch!(niter, t1.addcmul_(&t2, &t3));
                self.cache(result);
                dur
            }
            TorchCallInfo::AddCDiv_(info1, info2, info3) => {
                let mut t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let t3 = self.allocate(info3);
                let (result, dur) = estimate_torch!(niter, t1.addcdiv_(&t2, &t3));
                self.cache(result);
                dur
            }
            TorchCallInfo::ForeachAddCMul_(info_list1, info_list2, info_list3) => {
                let t_list2 = self.allocate_list(info_list2);
                let t_list3 = self.allocate_list(info_list3);
                let t_list1 = self.allocate_list(info_list1);
                let (_, dur) = estimate_torch!(
                    niter,
                    Tensor::f_internal_foreach_addcmul_(&t_list1, &t_list2, &t_list3, 2)
                );
                dur
            }
            TorchCallInfo::ForeachAddCDiv_(info_list1, info_list2, info_list3) => {
                let t_list2 = self.allocate_list(info_list2);
                let t_list3 = self.allocate_list(info_list3);
                let t_list1 = self.allocate_list(info_list1);
                let (_, dur) = estimate_torch!(
                    niter,
                    Tensor::f_internal_foreach_addcdiv_(&t_list1, &t_list2, &t_list3, 2)
                );
                dur
            }
            TorchCallInfo::MaskedFill(info, mask_info) => {
                let t = self.allocate(info);
                let mask = self.allocate(mask_info);
                let (result, dur) = estimate_torch!(niter, t.masked_fill(&mask, -10000.0));
                self.cache(result);
                dur
            }
            TorchCallInfo::MaskedFill_(info, mask_info) => {
                let mut t = self.allocate(info);
                let mask = self.allocate(mask_info);
                let (result, dur) = estimate_torch!(niter, t.masked_fill_(&mask, -10000.0));
                self.cache(result);
                dur
            }
            TorchCallInfo::Dropout(info, p_millionths, train) => {
                let t = self.allocate(info);
                let (result, dur) =
                    estimate_torch!(niter, t.dropout(*p_millionths as f64 / 1_000_000.0, *train));
                self.cache(result);
                dur
            }
            TorchCallInfo::NativeDropout(info, p_millionths, train) => {
                let t = self.allocate(info);
                let ((result, mask), dur) = estimate_torch!(
                    niter,
                    t.native_dropout(*p_millionths as f64 / 1_000_000.0, *train)
                );
                self.cache(result);
                self.cache(mask);
                dur
            }
            TorchCallInfo::NativeDropoutBackward(grad_info, mask_info, scale_millionths) => {
                let grad = self.allocate(grad_info);
                let mask = self.allocate(mask_info);
                let (result, dur) = estimate_torch!(
                    niter,
                    Tensor::native_dropout_backward(
                        &grad,
                        &mask,
                        *scale_millionths as f64 / 1_000_000.0
                    )
                );
                self.cache(result);
                dur
            }
            TorchCallInfo::FusedDropout(info, p_millionths) => {
                let t = self.allocate(info);
                let ((result, mask), dur) = estimate_torch!(
                    niter,
                    t.internal_fused_dropout(*p_millionths as f64 / 1_000_000.0)
                );
                self.cache(result);
                self.cache(mask);
                dur
            }
            TorchCallInfo::Where(info1, info2, info3) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let t3 = self.allocate(info3);
                let (result, dur) = estimate_torch!(niter, t2.where_self(&t1, &t3));
                self.cache(result);
                dur
            }
            TorchCallInfo::WhereScalar(info1, info2) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(niter, t2.where_scalarother(&t1, 1));
                self.cache(result);
                dur
            }
            TorchCallInfo::Gelu(info) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.gelu("none"));
                self.cache(result);
                dur
            }
            TorchCallInfo::GeluBackward(grad_info, input_info) => {
                let grad = self.allocate(grad_info);
                let input = self.allocate(input_info);
                let (result, dur) = estimate_torch!(niter, input.gelu_backward(&grad, "none"));
                self.cache(result);
                dur
            }
            TorchCallInfo::Silu(info) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.silu());
                self.cache(result);
                dur
            }
            TorchCallInfo::SiluBackward(grad_info, input_info) => {
                let grad = self.allocate(grad_info);
                let input = self.allocate(input_info);
                let (result, dur) = estimate_torch!(niter, input.silu_backward(&grad));
                self.cache(result);
                dur
            }
            TorchCallInfo::ToCopyUpcast(info) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.totype(tch::Kind::Float));
                self.cache(result);
                dur
            }
            TorchCallInfo::LogSoftmax(info, dim) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.log_softmax(*dim, info.dtype));
                self.cache(result);
                dur
            }
            TorchCallInfo::LogSoftmaxBackward(info1, info2, dim) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(
                    niter,
                    Tensor::internal_log_softmax_backward_data(&t1, &t2, *dim, info1.dtype)
                );
                self.cache(result);
                dur
            }
            TorchCallInfo::CrossEntropyLoss(info, t_info) => {
                let t = self.allocate(info);
                let tgt = match kind_range(t_info.dtype) {
                    KindRange::Integer => {
                        allocate_class_index_target(info, t_info, Device::Cuda(0))
                    }
                    _ => self.allocate(t_info),
                };
                let (result, dur) = estimate_torch!(
                    niter,
                    t.cross_entropy_loss(&tgt, None::<Tensor>, tch::Reduction::Mean, -100, 0.0)
                );
                self.cache(result);
                dur
            }
            TorchCallInfo::NllLossBackward(info) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.zeros_like());
                self.cache(result);
                dur
            }
            TorchCallInfo::Mean(info) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.mean(info.dtype));
                self.cache(result);
                dur
            }
            TorchCallInfo::Sum(info) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.sum(info.dtype));
                self.cache(result);
                dur
            }
            TorchCallInfo::Sqrt(info) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.sqrt());
                self.cache(result);
                dur
            }
            TorchCallInfo::Rsqrt(info) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.rsqrt());
                self.cache(result);
                dur
            }
            TorchCallInfo::Neg(info) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.neg());
                self.cache(result);
                dur
            }
            TorchCallInfo::Cat(infos, dim) => {
                let tensors: Vec<_> = infos.iter().map(|info| self.allocate(info)).collect();
                let tensor_refs: Vec<_> = tensors.iter().collect();
                let (result, dur) = estimate_torch!(niter, Tensor::cat(&tensor_refs, *dim));
                self.cache(result);
                dur
            }
            TorchCallInfo::Softmax(info, dim) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.softmax(*dim, info.dtype));
                self.cache(result);
                dur
            }
            TorchCallInfo::SoftmaxBackward(info1, info2, dim) => {
                let t1 = self.allocate(info1);
                let t2 = self.allocate(info2);
                let (result, dur) = estimate_torch!(
                    niter,
                    Tensor::internal_softmax_backward_data(&t1, &t2, *dim, info1.dtype)
                );
                self.cache(result);
                dur
            }
            TorchCallInfo::ZerosLike(info) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.zeros_like());
                self.cache(result);
                dur
            }
            TorchCallInfo::ConvDType(info, kind) => {
                let t = self.allocate(info);
                let (result, dur) = estimate_torch!(niter, t.totype(*kind));
                self.cache(result);
                dur
            }
            TorchCallInfo::SDPA {
                q,
                k,
                v,
                mask,
                causal,
                gqa,
            } => {
                let t_q = self.allocate(q);
                let t_k = self.allocate(k);
                let t_v = self.allocate(v);
                let t_mask = mask.as_ref().map(|info| self.allocate(info));
                let (result, dur) = estimate_torch!(
                    niter,
                    Tensor::f_scaled_dot_product_attention(
                        &t_q,
                        &t_k,
                        &t_v,
                        t_mask.as_ref(),
                        0.0,
                        *causal,
                        None,
                        *gqa
                    )
                    .unwrap()
                );
                self.cache(result);
                dur
            }
            TorchCallInfo::SDPAEfficientBackward {
                q,
                k,
                v,
                bias,
                causal,
            } => {
                // Build the forward graph via autograd, then time only the backward.
                let t_q = self.allocate(q).detach().set_requires_grad(true);
                let t_k = self.allocate(k).detach().set_requires_grad(true);
                let t_v = self.allocate(v).detach().set_requires_grad(true);
                let t_bias = bias.as_ref().map(|info| self.allocate(info));
                let out = Tensor::scaled_dot_product_attention(
                    &t_q,
                    &t_k,
                    &t_v,
                    t_bias.as_ref(),
                    0.0,
                    *causal,
                    None,
                    false,
                )
                .sum(Kind::Float);
                let (_, dur) = estimate_torch!(
                    niter,
                    Tensor::run_backward(&[&out], &[&t_q, &t_k, &t_v], true, false)
                );
                dur
            }
            TorchCallInfo::SDPABackward {
                grad,
                q,
                k,
                v,
                out,
                logsumexp,
                max_q,
                max_k,
                causal,
            } => {
                let t_grad = self.allocate(grad);
                let t_q = self.allocate(q);
                let t_k = self.allocate(k);
                let t_v = self.allocate(v);
                let t_out = self.allocate(out);
                let t_logsumexp = self.allocate(logsumexp);
                let cum_seq = Tensor::new(); // should be undefined
                let philox_seed = Tensor::zeros(&[2], (Kind::UInt64, Device::Cuda(0)));
                let philox_offset = Tensor::zeros(&[1], (Kind::UInt64, Device::Cuda(0)));
                let (_, dur) = estimate_torch!(
                    niter,
                    Tensor::f_internal_scaled_dot_product_flash_attention_backward(
                        &t_grad,
                        &t_q,
                        &t_k,
                        &t_v,
                        &t_out,
                        &t_logsumexp,
                        &cum_seq,
                        &cum_seq,
                        *max_q,
                        *max_k,
                        0.0,
                        *causal,
                        &philox_seed,
                        &philox_offset,
                        None,
                    )
                    .unwrap()
                );
                self.cache(t_grad);
                self.cache(t_out);
                dur
            }
            TorchCallInfo::Conv2d {
                input,
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups,
            } => {
                let input = self.allocate(input);
                let weight = self.allocate(weight);
                let bias = bias.as_ref().map(|info| self.allocate(info));
                let (result, dur) = estimate_torch!(
                    niter,
                    input.conv2d(&weight, bias.as_ref(), stride, padding, dilation, *groups)
                );
                self.cache(result);
                dur
            }
            TorchCallInfo::Conv2dBackward {
                grad_output,
                input,
                weight,
                kernel,
                stride,
                padding,
            } => {
                let grad_bias = Tensor::empty(&[weight.shape[0]], (weight.dtype, Device::Cuda(0)));
                let input = self.allocate(input);
                let weight = self.allocate(weight);
                // let grad_input = input.empty_like();
                // let grad_weight = weight.empty_like();
                let grad_output = self.allocate(grad_output);

                let (_, dur) = estimate_torch!(
                    niter,
                    input.internal_slow_conv2d_backward(
                        &input,
                        &weight,
                        &grad_bias,
                        &grad_output,
                        &weight,
                        kernel,
                        stride,
                        padding
                    )
                );
                // self.cache(grad_input);
                // self.cache(grad_weight);
                self.cache(grad_bias);
                dur
            }
        }
    }

    pub fn estimate(&mut self, call: &TorchCallInfo) -> Duration {
        if let Some(value) = self.compute_cache.get(call) {
            *value
        } else {
            let duration = self.run(2, call);
            self.compute_cache.insert(call.clone(), duration);
            duration
        }
    }

    pub fn estimate_sequence(&mut self, calls: &[TorchCall]) -> Vec<Duration> {
        // self.tensor_cache.clear();
        let seq_hash = {
            let mut hasher = DefaultHasher::new();
            for call in calls {
                TorchCallInfo::hash(&call.info, &mut hasher);
            }
            hasher.finish()
        };
        if let Some(value) = self.sequence_cache.get(&seq_hash) {
            return value.clone();
        } else {
            let mut durs = Vec::new();
            for call in calls {
                durs.push(self.run(1, &call.info));
            }
            self.sequence_cache.insert(seq_hash, durs.clone());
            durs
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn estimator_without_cuda() -> TorchEstimator {
        TorchEstimator {
            tensor_cache: LruCache::new(NonZeroUsize::new(32).unwrap()),
            compute_cache: HashMap::new(),
            sequence_cache: BTreeMap::new(),
        }
    }

    fn assert_all_targets_in_range(target: &Tensor, n_classes: i64) {
        let numel: i64 = target.size().iter().product();
        for index in 0..numel {
            let value = target.int64_value(&[index]);
            assert!(
                (0..n_classes).contains(&value),
                "target {value} at index {index} is outside [0, {n_classes})"
            );
        }
    }

    #[test]
    fn class_index_targets_ignore_arbitrary_cached_integer_contents() {
        let input_info = TensorInfo {
            shape: vec![4, 3],
            dtype: Kind::Float,
        };
        let target_info = TensorInfo {
            shape: vec![4],
            dtype: Kind::Int64,
        };
        let mut estimator = estimator_without_cuda();
        let cached = Tensor::from_slice(&[-100_i64, 0, 3, 127]);
        estimator.tensor_cache.put((4, Kind::Int64), cached);

        let fresh_integer = estimator.allocate_on(&target_info, Device::Cpu);
        assert_ne!(fresh_integer.int64_value(&[0]), -100);
        assert!(fresh_integer.int64_value(&[3]) < 128);

        let target = allocate_class_index_target(&input_info, &target_info, Device::Cpu);
        assert_all_targets_in_range(&target, 3);
        assert_eq!(target.int64_value(&[0]), 0);
        assert_eq!(target.int64_value(&[3]), 0);
    }

    #[test]
    fn mb1_flattened_target_does_not_alias_same_numel_integer_cache_entry() {
        let sequence_length = 8192;
        let n_classes = 100_278;
        let input_info = TensorInfo {
            shape: vec![sequence_length, n_classes],
            dtype: Kind::BFloat16,
        };
        let target_info = TensorInfo {
            shape: vec![sequence_length],
            dtype: Kind::Int64,
        };
        let mut estimator = estimator_without_cuda();
        let unrelated = Tensor::full(&[1, sequence_length], n_classes, (Kind::Int64, Device::Cpu));
        estimator
            .tensor_cache
            .put((sequence_length, Kind::Int64), unrelated);

        let fresh_integer = estimator.allocate_on(&target_info, Device::Cpu);
        assert_ne!(fresh_integer.int64_value(&[0]), n_classes);

        let target = allocate_class_index_target(&input_info, &target_info, Device::Cpu);
        assert_all_targets_in_range(&target, n_classes);
        assert_eq!(target.size(), target_info.shape);
    }

    #[test]
    fn tensor_cache_is_float_only() {
        for dtype in [Kind::Bool, Kind::Int, Kind::Int64] {
            assert_eq!(
                tensor_cache_key(&TensorInfo {
                    shape: vec![1, 8192],
                    dtype,
                }),
                None
            );
        }
        assert_eq!(
            tensor_cache_key(&TensorInfo {
                shape: vec![1, 8192],
                dtype: Kind::BFloat16,
            }),
            Some((8192, Kind::BFloat16))
        );
    }
}
