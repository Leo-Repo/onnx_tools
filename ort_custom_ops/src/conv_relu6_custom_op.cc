#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>


namespace {

std::vector<int64_t> GetAttrInts(const OrtApi& api, const OrtKernelInfo* info, const char* key,
                                 const std::vector<int64_t>& defaults) {
  size_t size = 0;
  OrtStatus* status = api.KernelInfoGetAttributeArray_int64(info, key, nullptr, &size);
  if (status != nullptr) {
    api.ReleaseStatus(status);
    return defaults;
  }
  std::vector<int64_t> vals(size);
  status = api.KernelInfoGetAttributeArray_int64(info, key, vals.data(), &size);
  if (status != nullptr) {
    api.ReleaseStatus(status);
    return defaults;
  }
  return vals;
}

int64_t GetAttrInt(const OrtApi& api, const OrtKernelInfo* info, const char* key, int64_t default_val) {
  int64_t val = default_val;
  OrtStatus* status = api.KernelInfoGetAttribute_int64(info, key, &val);
  if (status != nullptr) {
    api.ReleaseStatus(status);
    return default_val;
  }
  return val;
}

std::string GetAttrString(const OrtApi& api, const OrtKernelInfo* info, const char* key, const std::string& defaults) {
  size_t size = 0;
  OrtStatus* status = api.KernelInfoGetAttribute_string(info, key, nullptr, &size);
  if (status != nullptr) {
    api.ReleaseStatus(status);
    return defaults;
  }
  std::string out(size, '\0');
  status = api.KernelInfoGetAttribute_string(info, key, out.data(), &size);
  if (status != nullptr) {
    api.ReleaseStatus(status);
    return defaults;
  }
  if (!out.empty() && out.back() == '\0') {
    out.pop_back();
  }
  return out;
}

struct ConvRelu6Kernel {explicit ConvRelu6Kernel(const OrtApi& api, const OrtKernelInfo* info) : api_(api) {
    pads_ = GetAttrInts(api_, info, "Conv0_pads", {0, 0, 0, 0});
    strides_ = GetAttrInts(api_, info, "Conv0_strides", {1, 1});
    dilations_ = GetAttrInts(api_, info, "Conv0_dilations", {1, 1});
    group_ = static_cast<int>(GetAttrInt(api_, info, "Conv0_group", 1));
    auto_pad_ = GetAttrString(api_, info, "Conv0_auto_pad", "NOTSET");
  }

void Compute(OrtKernelContext* ctx) {
    const OrtValue* x = nullptr;
    const OrtValue* w = nullptr;
    const OrtValue* b = nullptr;
    OrtStatus* status = api_.KernelContext_GetInput(ctx, 0, &x);
    if (status != nullptr) {
      api_.ReleaseStatus(status);
      throw std::runtime_error("KernelContext_GetInput(0) failed.");
    }
    status = api_.KernelContext_GetInput(ctx, 1, &w);
    if (status != nullptr) {
      api_.ReleaseStatus(status);
      throw std::runtime_error("KernelContext_GetInput(1) failed.");
    }
    status = api_.KernelContext_GetInput(ctx, 2, &b);
    if (status != nullptr) {
      api_.ReleaseStatus(status);
      throw std::runtime_error("KernelContext_GetInput(2) failed.");
    }

    const auto x_dims = GetDims(x);
    const auto w_dims = GetDims(w);
    if (x_dims.size() != 4 || w_dims.size() != 4) {
      throw std::runtime_error("Conv_ReLU6 currently supports 2D NCHW conv only.");
    }

    const int64_t n = x_dims[0];
    const int64_t in_c = x_dims[1];
    const int64_t in_h = x_dims[2];
    const int64_t in_w = x_dims[3];
    const int64_t out_c = w_dims[0];
    const int64_t ker_c = w_dims[1];
    const int64_t ker_h = w_dims[2];
    const int64_t ker_w = w_dims[3];

    if (group_ <= 0 || out_c % group_ != 0) {
      throw std::runtime_error("Invalid group/out channel configuration.");
    }

    int64_t pad_top = pads_.size() > 0 ? pads_[0] : 0;
    int64_t pad_left = pads_.size() > 1 ? pads_[1] : 0;
    int64_t pad_bottom = pads_.size() > 2 ? pads_[2] : pad_top;
    int64_t pad_right = pads_.size() > 3 ? pads_[3] : pad_left;
    const int64_t stride_h = strides_.size() > 0 ? strides_[0] : 1;
    const int64_t stride_w = strides_.size() > 1 ? strides_[1] : stride_h;
    const int64_t dil_h = dilations_.size() > 0 ? dilations_[0] : 1;
    const int64_t dil_w = dilations_.size() > 1 ? dilations_[1] : dil_h;

    int64_t out_h = 0;
    int64_t out_w = 0;
    if (auto_pad_ == "SAME_UPPER" || auto_pad_ == "SAME_LOWER") {
      out_h = (in_h + stride_h - 1) / stride_h;
      out_w = (in_w + stride_w - 1) / stride_w;
      const int64_t need_h = std::max<int64_t>(0, (out_h - 1) * stride_h + (ker_h - 1) * dil_h + 1 - in_h);
      const int64_t need_w = std::max<int64_t>(0, (out_w - 1) * stride_w + (ker_w - 1) * dil_w + 1 - in_w);
      if (auto_pad_ == "SAME_UPPER") {
        pad_top = need_h / 2;
        pad_bottom = need_h - pad_top;
        pad_left = need_w / 2;
        pad_right = need_w - pad_left;
      } else {
        pad_bottom = need_h / 2;
        pad_top = need_h - pad_bottom;
        pad_right = need_w / 2;
        pad_left = need_w - pad_right;
      }
    } else {
      out_h = (in_h + pad_top + pad_bottom - dil_h * (ker_h - 1) - 1) / stride_h + 1;
      out_w = (in_w + pad_left + pad_right - dil_w * (ker_w - 1) - 1) / stride_w + 1;
    }

    const std::vector<int64_t> y_shape{n, out_c, out_h, out_w};
    OrtValue* y = nullptr;
    status = api_.KernelContext_GetOutput(ctx, 0, y_shape.data(), y_shape.size(), &y);
    if (status != nullptr) {
      api_.ReleaseStatus(status);
      throw std::runtime_error("KernelContext_GetOutput failed.");
    }

    const float* x_data = nullptr;
    const float* w_data = nullptr;
    float* y_data = nullptr;
    auto* b_data = static_cast<const float*>(nullptr);
    if (api_.GetTensorMutableData(const_cast<OrtValue*>(x), reinterpret_cast<void**>(const_cast<float**>(&x_data))) != nullptr) {
      throw std::runtime_error("Failed to get X tensor data.");
    }
    if (api_.GetTensorMutableData(const_cast<OrtValue*>(w), reinterpret_cast<void**>(const_cast<float**>(&w_data))) != nullptr) {
      throw std::runtime_error("Failed to get W tensor data.");
    }
    if (b != nullptr) {
      if (api_.GetTensorMutableData(const_cast<OrtValue*>(b), reinterpret_cast<void**>(const_cast<float**>(&b_data))) != nullptr) {
        throw std::runtime_error("Failed to get B tensor data.");
      }
    }
    if (api_.GetTensorMutableData(y, reinterpret_cast<void**>(&y_data)) != nullptr) {
      throw std::runtime_error("Failed to get Y tensor data.");
    }

    const int64_t out_c_per_group = out_c / group_;
    const int64_t in_c_per_group = ker_c;

    for (int64_t bn = 0; bn < n; ++bn) {
      for (int64_t oc = 0; oc < out_c; ++oc) {
        const int64_t group_idx = oc / out_c_per_group;
        const int64_t ic_offset = group_idx * in_c_per_group;
        for (int64_t oh = 0; oh < out_h; ++oh) {
          for (int64_t ow = 0; ow < out_w; ++ow) {
            float acc = b_data ? b_data[oc] : 0.0f;
            for (int64_t ic = 0; ic < in_c_per_group; ++ic) {
              for (int64_t kh = 0; kh < ker_h; ++kh) {
                for (int64_t kw = 0; kw < ker_w; ++kw) {
                  const int64_t ih = oh * stride_h + kh * dil_h - pad_top;
                  const int64_t iw = ow * stride_w + kw * dil_w - pad_left;
                  if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) {
                    continue;
                  }
                  const int64_t in_idx = ((bn * in_c + (ic_offset + ic)) * in_h + ih) * in_w + iw;
                  const int64_t w_idx = ((oc * ker_c + ic) * ker_h + kh) * ker_w + kw;
                  acc += x_data[in_idx] * w_data[w_idx];
                }
              }
            }
            acc = std::min(6.0f, std::max(0.0f, acc));
            const int64_t out_idx = ((bn * out_c + oc) * out_h + oh) * out_w + ow;
            y_data[out_idx] = acc;
          }
        }
      }
    }
  }

  std::vector<int64_t> GetDims(const OrtValue* t) const {
    OrtTensorTypeAndShapeInfo* shape_info = nullptr;
    OrtStatus* status = api_.GetTensorTypeAndShape(t, &shape_info);
    if (status != nullptr) {
      api_.ReleaseStatus(status);
      throw std::runtime_error("GetTensorTypeAndShape failed.");
    }

    size_t dim_count = 0;
    status = api_.GetDimensionsCount(shape_info, &dim_count);
    if (status != nullptr) {
      api_.ReleaseTensorTypeAndShapeInfo(shape_info);
      api_.ReleaseStatus(status);
      throw std::runtime_error("GetDimensionsCount failed.");
    }

    std::vector<int64_t> dims(dim_count);
    status = api_.GetDimensions(shape_info, dims.data(), dim_count);
    api_.ReleaseTensorTypeAndShapeInfo(shape_info);
    if (status != nullptr) {
      api_.ReleaseStatus(status);
      throw std::runtime_error("GetDimensions failed.");
    }
    return dims;
  }

  const OrtApi& api_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> dilations_;
  int group_ = 1;
  std::string auto_pad_ = "NOTSET";
};

struct ConvRelu6Op : Ort::CustomOpBase<ConvRelu6Op, ConvRelu6Kernel> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const { return new ConvRelu6Kernel(api, info); }
  const char* GetName() const { return "Conv_ReLU6"; }
  const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  const char* GetDomain() const { return "ai.custom"; }
  size_t GetInputTypeCount() const { return 5; }   // X, W, B, clip_min, clip_max
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
};

ConvRelu6Op g_conv_relu6_op;

}  // namespace

extern "C" __declspec(dllexport) OrtStatus* ORT_API_CALL RegisterCustomOps(
    OrtSessionOptions* options, const OrtApiBase* api_base) {
  const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
  auto* ort_options = reinterpret_cast<OrtSessionOptions*>(options);
  Ort::UnownedSessionOptions session_options{ort_options};
  static Ort::CustomOpDomain domain{"ai.custom"};
  domain.Add(&g_conv_relu6_op);
  session_options.Add(domain);
  return nullptr;
}
