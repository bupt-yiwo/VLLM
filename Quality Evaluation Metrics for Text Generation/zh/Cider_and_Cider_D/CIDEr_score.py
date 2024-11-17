from utils.cider import compute_cider_interface

gts = {
    'image1': ['a b c d']
}
res = [
    {'image_id': 'image1', 'caption': ['a b c d']}
]

score = compute_cider_interface(gts, res)
print("CIDEr Score:", score)
