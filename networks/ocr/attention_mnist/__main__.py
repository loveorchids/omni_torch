import os, sys
sys.path.append(os.path.expanduser("~/Documents/omni_research/"))
import networks.ocr.attention_mnist.attention_mnist as attention_mnist

if __name__ == "__main__":
    attention_mnist.main()