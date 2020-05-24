在虚拟机中安装安装失败，正在研究中。。。

安装anaconda    //使用清华镜像，镜像站:https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/
conda creat 创建环境
https://www.mindspore.cn/versions 下载ubuntu环境安装包
共享到虚拟机 执行安装

配套软件包依赖配置
安装Ascend 910 AI处理器配套软件包（对应版本Atlas T 1.1.T107）提供的whl包，whl包随配套软件包发布，升级配套软件包之后需要重新安装。
//此步骤报错  或许非Ascend AI处理器不需要此whl开发包？ 有待联系huawei service
pip install /usr/local/Ascend/fwkacllib/lib64/topi-{version}-py3-none-any.whl
pip install /usr/local/Ascend/fwkacllib/lib64/te-{version}-py3-none-any.whl
pip install /usr/local/Ascend/fwkacllib/lib64/hccl-{version}-py3-none-any.whl

//TODO
配置环境变量
# control log level. 0-DEBUG, 1-INFO, 2-WARNING, 3-ERROR, default level is WARNING.
export GLOG_v=2
# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package
# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/add-ons/:${LOCAL_ASCEND}/fwkacllib/lib64:${LD_LIBRARY_PATH}
# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/opp/op_impl/built-in/ai_core/tbe # TBE operator implementation tool path
export PATH=${LOCAL_ASCEND}/fwkacllib/ccec_compiler/bin/:${PATH} # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH} # Python library that TBE implementation depends on


安装验证
import numpy as np
from mindspore import Tensor
from mindspore.ops import functional as F
import mindspore.context as context

context.set_context(device_target="Ascend")
x = Tensor(np.ones([1,3,3,4]).astype(np.float32))
y = Tensor(np.ones([1,3,3,4]).astype(np.float32))
print(F.tensor_add(x, y))
