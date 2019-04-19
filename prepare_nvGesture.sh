# mkdir annotation_nvGesture
# python utils/nv_prepare.py training trainlistall.txt all
# python utils/nv_prepare.py training trainlistall_but_None.txt all_but_None
# python utils/nv_prepare.py training trainlistbinary.txt binary
# python utils/nv_prepare.py validation vallistall.txt all
# python utils/nv_prepare.py validation vallistall_but_None.txt all_but_None
# python utils/nv_prepare.py validation vallistbinary.txt binaryinary

python utils/nv_json.py 'annotation_nvGesture' all
python utils/nv_json.py 'annotation_nvGesture' all_but_None
python utils/nv_json.py 'annotation_nvGesture' binary