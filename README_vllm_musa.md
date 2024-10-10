# vllm_musa

摩尔线程致力于构建完善好用的国产GPU应用生态，自主研发了MUSA架构及软件平台。vllm项目是业界广泛使用的大语言模型的推理和服务引擎，使用CUDA/ROCm提供GPU加速能力。为了方便摩尔线程GPU用户使用vllm框架，我们发起vllm_musa开源项目为vllm提供MUSA加速，让用户可释放摩尔线程GPU的澎湃算力。

现有的vllm代码不支持摩尔线程GPU作为后端，因此我们新增了MUSA设备后端。vllm_musa接口与官方接口一致，用户无需改动业务代码，开箱即用。

MUSA的一大优势是CUDA兼容，通过musify工具，我们可以快速将官方代码porting至MUSA软件栈，用户可以根据文档自行升级vllm版本并适配MUSA软件栈。

## 依赖

- musa_toolkit dev3.0.0
- pytorch v2.2.0
- [torch_musa v2.0.0](https://github.com/MooreThreads/torch_musa)
- triton >= v2.2.0
- ray >= 2.9
- vllm v0.4.2

## 使用
### 编译
运行 `bash build_musa.sh`
### 测试示例
```
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, LlamaForCausalLM
import transformers
import time
import torch
import torch_musa


model_path = <path_to_llm_model>

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model=model_path, trust_remote_code=True, device="musa")

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

```

## Porting

当前仓库porting自vllm v0.4.2版本。如果用户希望使用更高版本的vllm，只需要运行`musa_porting.py`将原生CUDA代码适配到MUSA代码即可。当然随着vllm的迭代可能会有些代码成为漏网之鱼，没有porting成功，用户可自行修改`musa_porting.py`文件中的文本替换规则。从而发挥MUSA强大的CUDA兼容能力。

### 步骤
1. 运行 `python musa_porting.py`
2. 将`CMakeLists.txt`中需要编译的文件后缀从`.cu`修改为`.mu`
3. 编译运行vllm_musa

## 贡献

欢迎广大用户及开发者使用、反馈，助力vllm_musa功能及性能持续完善。

社区共建，期待广大开发者与我们一道，共同打造MUSA软件生态。我们将陆续推出一系列开源软件MUSA加速项目。