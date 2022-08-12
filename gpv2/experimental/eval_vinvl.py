import json
from collections import defaultdict
from os.path import join, dirname, expanduser, exists

from gpv2 import file_paths
from gpv2.data.dataset import Task, LocalizationExample
from gpv2.data.dce_dataset import DceDataset
import pickle

from gpv2.model.model import GPVExampleOutput
import numpy as np

from gpv2.train.evaluator import LocalizationEvaluator
from gpv2.utils import py_utils
from collections import Counter


LABEL_MAP = {
  "human face": ["face", "head"],
  "human arm": ["arm"],
  "human hand": ["hand"],
  "human ear": ["ear"],
  "human hair": ["hair"],
  "human head": ["head", "face"],
  "human nose": ["nose"],
  "human leg": ["leg"],
  "human eye": ["eye"],
  "human foot": ["foot"],
  "human mouth": ["mouth"],
  "tin can": ["can"],
  "vehicle registration plate": ["license plate"],
  "flowerpot": ["flower pot"],
  "high heels": ["shoe"],
  "swim cap": ["cap"],
  "wine glass": ["glass"],
  "jeans": ["jean"],
  "shorts": ["short"],
  "mule": ["donkey"],
  "table tennis racket": ["racket"],
  "miniskirt": ["skirt"],
  "headphones": ["headphone"],
  "microwave oven": ["oven"],
  "gas stove": ["stove"],
  "doughnut": ["donut"],
  "stairs": ["stair"],
  "earrings": ["earring"],
  "bell pepper": ["pepper"],
  "chopsticks": ["chopstick"],
  "window blind": ["blind"],
  "tablet computer": ["tablet"],
  "picnic basket": ["basket"],
  "wardrobe": ["closet"],
  "drinking straw": ["straw"],
  "horizontal bar": ["bar"],
  "corded phone": ["telephone"],
  "golf cart": ["cart"],
  "plastic bag": ["bag"],
  "salt and pepper shakers": ["pepper shaker", "shaker"],
  "adhesive tape": ["tape"],
  "training bench": ["bench"]
}

NO_MATCH  = [
  "screwdriver", "spice rack", "binoculars", "honeycomb", "wrench",
  "ruler", "missile", "cannon", "rocket", "plugs and sockets",
  "dumbbell", "syringe", "eraser", "axe", "bottle opener",
  "band-aid", "chime",
  "rays and skates", "roller skates", "infant bed", "bowling equipment",
  "bow and arrow", "beehive", "balance beam", "wine rack",
  "submarine", "fax", "treadmill", "seat belt", "pencil case", "stretcher",
  "chisel", "crutch", "chainsaw", "cat furniture", "stethoscope",
  "face powder", "drill", "punching bag", "diaper",
]

for x in NO_MATCH:
  LABEL_MAP[x] = []

SUPER_TYPES = {
  "animal": ["fox", "leopard", "lizard", "jaguar", "sea turtle", "tortoise", "crocodile",
             "snail", "hedgehog", "porcupine", "camel", "cheetah", "raccoon", "oyster",
             "red panda", "hippopotamus", "skunk", "isopod", "tick",
             "kangaroo", "lobster", "ladybug", "caterpillar", "brown bear", "ant",
             "rhinoceros", "scorpion", "dragonfly", "centipede", "koala", "worm",
             "squirrel", "snake", "spider", "bee", "alpaca", "otter", "sea lion", "hamster"],
  "bird": ["raven", "blue jay", "falcon", "canary", "sparrow", "woodpecker", "magpie"],
  "flower": ["lavender", "lily"],
  "light": ["torch", "flashlight"],
  "instrument": ["violin", "trumpet", "organ", "saxophone", "cello", "banjo",
                 "harmonica",
                 "harpsichord", "accordion", "harp", "trombone", "flute", "oboe"],
  "tree": ["maple", "willow"],
  "cooker": ["pressure cooker", "slow cooker"],
  "fish": ["shark", "seahorse", "starfish", "goldfish", "jellyfish", "squid"],
  "machine": ["coffeemaker", "sewing machine", "cassette deck",
              "indoor rower", "hand dryer"],
  "food": ["guacamole", "sushi", "taco", "tart", "pretzel", "popcorn"],
  "gun": ["handgun", "shotgun", "rifle"],
  "vehicle": ["snowplow", "gondola", "segway", "snowmobile", "unicycle"],
  "ball": ["volleyball", "golf ball", "rugby ball", "cricket ball"],
  "boat": ["jet ski", "barge"],
  "fruit": ["cantaloupe", "pomegranate", "winter melon", "fig"],
  "clock": ["digital clock", "wall clock"],
  "helmet": ["football helmet", "bicycle helmet"],
  "pan": ["waffle iron", "frying pan"]
}

SUBTYPES = {
  "beaker": ["glass"],
  "facial tissue holder": ["box"],
  "food processor": ["blender", "mixer"],
  "dog bed": ["pillow"],
  "waffle": ["dessert", "food"],
  "tree house": ["house", "building", "structure"],
  "perfume": ["bottle"],
  "tiara": ["crown"],
  "burrito": ["sandwich", "food"],
  "dice": ["block", "toy"],
  "lynx": ["cat", "animal"],
  "french horn": ["horn"],
  "dolphin": ["fish", "animal"],
  "whale": ["fish", "animal"],
  "dustbin": ["trash can"],
  "swimwear": ["bikini", "short"],
  "brassiere": ["bikini"],
  "mobile phone": ["phone"],
  "bathroom cabinet": ["medicine cabinet", "cabinet"],
  "wok": ["skillet", "pan"],
  "convenience store": ["building"],
  "wood-burning stove": ["stove"],
  "houseplant": ["plant"],
  "limousine": ["car"],
  "billiard table": ["table"],
  "sports uniform": ["uniform"],
  "cocktail": ["drink"],
  "bathtub": ["tub"],
  "jacuzzi": ["tub", "bathtub"],
}

for k, v in SUPER_TYPES.items():
  for x in v:
    SUBTYPES[x] = [k]

LABEL_MAP.update(SUBTYPES)


def main(split="test"):
  label_map = join(file_paths.DATA_DIR, "vinvl", "VG-SGG-dicts-vgoi6-clipped.json")
  with open(label_map) as f:
    label_map = json.load(f)
  id_to_label = {int(k): v for k, v in label_map["idx_to_label"].items()}
  label_to_idx = {v: k for k, v in id_to_label.items()}
  for k, v in LABEL_MAP.items():
    for label in v:
      assert label in label_to_idx

  dataset = DceDataset(Task.LOCALIZATION, split)
  examples = dataset.load()

  with open(join(expanduser("~/"), "Desktop", f"out-{dataset.part}.pkl"), "rb") as f:
    results = pickle.load(f)

  predictions = {}
  for ex in examples:
    pred = results[ex.gpv_id]
    cats = LABEL_MAP.get(ex.category, [ex.category])
    ixs = []
    for cat in cats:
      if cat in pred["labels"]:
        ixs = [i for i, c in enumerate(pred["labels"]) if c == cat]
        break

    score = np.array(pred["scores"])
    score[ixs] += 1.0
    predictions[ex.gpv_id] = GPVExampleOutput(pred["boxes"], score, None, None)

  print(LocalizationEvaluator().evaluate(examples, predictions))

def check_map(split):
  label_map = join(file_paths.DATA_DIR, "vinvl", "VG-SGG-dicts-vgoi6-clipped.json")
  with open(label_map) as f:
    label_map = json.load(f)
  id_to_label = {int(k): v for k, v in label_map["idx_to_label"].items()}
  label_to_idx = {v: k for k, v in id_to_label.items()}
  for k, v in LABEL_MAP.items():
    for label in v:
      assert label in label_to_idx

  dataset = DceDataset(Task.LOCALIZATION, split)
  examples = dataset.load()

  with open(join(expanduser("~/"), "Desktop", f"out-{dataset.part}.pkl"), "rb") as f:
    results = pickle.load(f)

  grouped = defaultdict(list)
  for ex in examples:
    pred = results[ex.gpv_id]
    cats = LABEL_MAP.get(ex.category, [ex.category])
    if ex.category not in LABEL_MAP and ex.category not in label_to_idx:
      grouped[ex.category].append(pred["labels"])

  print(len(grouped))
  for k in sorted(grouped, key=lambda x: -len(grouped[x]))[:10]:
    print(k)
    print(Counter(py_utils.flatten_list(grouped[k])))

if __name__ == '__main__':
  main("test")