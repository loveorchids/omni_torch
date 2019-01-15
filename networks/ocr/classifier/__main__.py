import os, sys
sys.path.append(os.path.expanduser("~/Documents/omni_research/"))
import networks.ocr.classifier.classifier as classifier

if __name__ == "__main__":
    classifier.main()