### Vid2Vid

#### Environment
CUDA=10.1
pytorch=1.4.0


Vid2Vid의 공식 코드가 너무 복잡하여 디폴트로 사용되는 것들만 정리함. FlowNet을 필수로 설치해야 함을 명심하자. 또한 FlowNetv2때문에 위의 환경을 꼭 맞춰줘야 한다. FlowNet설정은 아래를 참고하자.

1. git clone the flownet2-pytorch repository directly into the models folder in vid2vid to flownet2_pytorch (recommended)
이 코드를 git clone하면 이미 위의 수정사항을 반영했기 때문에 안해도 된다.

2. [링크](https://drive.google.com/file/d/1u5WFXBWGyvWmeyfs5bFdnHweqC4dOTLP/view?usp=sharing)에서 FlowNet2모델을 다운 받고, models/flownet2_pytorch에 넣는다.

3. Rebuild flownet2 after making those changes with bash install.sh inside the models/flownet2_pytorch folder. It should now compile successfully with PyTorch 1.0.0.