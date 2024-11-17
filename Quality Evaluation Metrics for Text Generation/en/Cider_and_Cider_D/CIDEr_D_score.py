from utils.CIDEr_D import CiderD

gts = {
    "image1": ["a man is riding a horse", "a person on a horse"],
    "image2": ["a cat is sleeping", "a feline is resting"]
}
res = [
    {"image_id": "image1", "caption": ["a man rides a horse"]},
    {"image_id": "image2", "caption": ["a cat sleeps"]}
]

cider = CiderD()
score, scores = cider.compute_score(gts, res)
print(f"CIDEr-D Score: {score}")
print(f"Detailed Scores: {scores}")