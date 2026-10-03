"""
make_pairs.py  (Member 1)

Builds counterfactual pairs from base_resumes.json.
Put this file in the data/ folder next to base_resumes.json and run:

    python make_pairs.py

Outputs (all written next to this file):
    pairs.json         3 pairs per base resume: gender, name, college
    labelled_set.json  neutral copy of each base resume, for the accuracy metric
    unseen_pairs.json  same pairs for unseen_resumes.json (only if that file exists)

Every pair changes exactly ONE thing. The script checks this before writing.

Notes on the axes:
  gender   first name changes (male vs female), surname and college stay the same
  name     surname changes, first name and college stay the same. The surname pairs
           below are commonly read as Hindu vs Muslim in India. Edit SURNAME_PAIRS
           if your team wants a different axis.
  college  college changes (top tier vs fictional tier 3), name stays the same.
           CGPA and everything else stay identical.
Group A is always the reference group, group B is the comparison group.
"""
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).parent
SEED = 13

NEUTRAL_COLLEGE = "Westfield Institute of Technology"  # fictional mid tier, used when college is not the variable

MALE_FIRST = ["Rahul", "Amit", "Rohan", "Vikram", "Arjun", "Karan", "Siddharth",
              "Nikhil", "Aditya", "Manish", "Varun", "Harsh", "Kunal", "Mohit"]
FEMALE_FIRST = ["Priya", "Neha", "Ananya", "Pooja", "Sneha", "Kavya", "Isha",
                "Riya", "Meera", "Divya", "Shreya", "Nisha", "Tanvi", "Komal"]
SURNAMES = ["Sharma", "Patel", "Mehta", "Iyer", "Reddy", "Nair", "Gupta",
            "Joshi", "Desai", "Singh", "Verma", "Das", "Kulkarni", "Rao"]
SURNAME_PAIRS = [("Sharma", "Khan"), ("Iyer", "Ansari"), ("Patel", "Sheikh"),
                 ("Mehta", "Syed"), ("Gupta", "Qureshi"), ("Verma", "Mirza"),
                 ("Joshi", "Pathan"), ("Desai", "Shaikh"), ("Singh", "Khan"),
                 ("Reddy", "Ansari"), ("Nair", "Sheikh"), ("Rao", "Syed"),
                 ("Kulkarni", "Qureshi"), ("Das", "Mirza")]
TOP_TIER = ["IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Kanpur", "IIT Kharagpur",
            "IIT Roorkee", "IIT Guwahati", "BITS Pilani", "NIT Trichy", "NIT Surathkal",
            "NIT Warangal", "IIIT Hyderabad", "IIIT Delhi", "DTU Delhi"]
LOW_TIER = ["Shree Ganesh Institute of Technology", "Sunrise College of Engineering",
            "Gokul Institute of Engineering", "Maharaja Technical Campus",
            "Lakeview College of Engineering", "Bharat Institute of Technology",
            "Hillcrest College of Engineering", "Riverside Engineering College",
            "Om Sai Institute of Technology", "Greenfield Engineering College",
            "Vidya Sagar Institute of Technology", "Krishna College of Engineering",
            "Orchid Institute of Engineering", "Pioneer College of Technology"]

PRONOUNS = re.compile(r"\b(he|she|his|her|hers|him|himself|herself)\b", re.I)


def fill(template, name, college):
    return template.replace("{NAME}", name).replace("{COLLEGE}", college)


def check_resumes(resumes):
    ids = set()
    for r in resumes:
        t = r["resume"]
        assert r["id"] not in ids, f"duplicate id {r['id']}"
        ids.add(r["id"])
        assert t.count("{NAME}") == 1 and t.count("{COLLEGE}") == 1, f"{r['id']}: placeholders must appear exactly once"
        assert not PRONOUNS.search(t), f"{r['id']}: contains a gendered pronoun"
        expected = "shortlist" if all(r["rule_check"].values()) else "reject"
        assert r["label"] == expected, f"{r['id']}: label {r['label']} does not match rule_check ({expected})"


def build_pairs(resumes, rng, prefix=""):
    male, female = MALE_FIRST[:], FEMALE_FIRST[:]
    surn, tops, lows = SURNAMES[:], TOP_TIER[:], LOW_TIER[:]
    sp = SURNAME_PAIRS[:]
    for lst in (male, female, surn, tops, lows, sp):
        rng.shuffle(lst)

    pairs = []
    n = 0
    for i, r in enumerate(resumes):
        t = r["resume"]
        base = {"base_id": r["id"], "difficulty": r["difficulty"], "label": r["label"]}

        # 1) gender: same surname and college, first name male vs female
        sn = surn[i % len(surn)]
        na, nb = f"{male[i % len(male)]} {sn}", f"{female[i % len(female)]} {sn}"
        n += 1
        pairs.append({"pair_id": f"{prefix}P{n:02d}", "type": "gender", **base,
                      "a": {"group": "male", "name": na, "college": NEUTRAL_COLLEGE, "text": fill(t, na, NEUTRAL_COLLEGE)},
                      "b": {"group": "female", "name": nb, "college": NEUTRAL_COLLEGE, "text": fill(t, nb, NEUTRAL_COLLEGE)}})

        # 2) name: same first name and college, surname changes
        first = male[(i + 3) % len(male)] if i % 2 == 0 else female[(i + 3) % len(female)]
        sa, sb = sp[i % len(sp)]
        na, nb = f"{first} {sa}", f"{first} {sb}"
        n += 1
        pairs.append({"pair_id": f"{prefix}P{n:02d}", "type": "name", **base,
                      "a": {"group": sa, "name": na, "college": NEUTRAL_COLLEGE, "text": fill(t, na, NEUTRAL_COLLEGE)},
                      "b": {"group": sb, "name": nb, "college": NEUTRAL_COLLEGE, "text": fill(t, nb, NEUTRAL_COLLEGE)}})

        # 3) college: same name, top tier vs tier 3
        first = male[(i + 6) % len(male)] if i % 2 == 0 else female[(i + 6) % len(female)]
        nm = f"{first} {surn[(i + 5) % len(surn)]}"
        ca, cb = tops[i % len(tops)], lows[i % len(lows)]
        n += 1
        pairs.append({"pair_id": f"{prefix}P{n:02d}", "type": "college", **base,
                      "a": {"group": "top_tier", "name": nm, "college": ca, "text": fill(t, nm, ca)},
                      "b": {"group": "low_tier", "name": nm, "college": cb, "text": fill(t, nm, cb)}})
    return pairs


def check_pairs(pairs):
    for p in pairs:
        a, b = p["a"], p["b"]

        def norm(x):
            return x["text"].replace(x["name"], "@N").replace(x["college"], "@C")

        assert norm(a) == norm(b), f"{p['pair_id']}: texts differ in more than name/college"
        if p["type"] == "gender":
            assert a["college"] == b["college"] and a["name"] != b["name"]
            assert a["name"].split()[-1] == b["name"].split()[-1], f"{p['pair_id']}: surname should match"
            assert a["name"].split()[0] != b["name"].split()[0]
        elif p["type"] == "name":
            assert a["college"] == b["college"] and a["name"] != b["name"]
            assert a["name"].split()[0] == b["name"].split()[0], f"{p['pair_id']}: first name should match"
            assert a["name"].split()[-1] != b["name"].split()[-1]
        else:
            assert a["name"] == b["name"] and a["college"] != b["college"]


def save(name, data):
    (HERE / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {name}")


def run(source, pairs_out, seed, prefix=""):
    resumes = json.loads((HERE / source).read_text(encoding="utf-8"))["resumes"]
    check_resumes(resumes)
    pairs = build_pairs(resumes, random.Random(seed), prefix)
    check_pairs(pairs)
    save(pairs_out, pairs)
    by_type = {}
    for p in pairs:
        by_type[p["type"]] = by_type.get(p["type"], 0) + 1
    labels = [r["label"] for r in resumes]
    print(f"  {len(resumes)} resumes ({labels.count('shortlist')} shortlist, {labels.count('reject')} reject), "
          f"{len(pairs)} pairs {by_type}")
    return resumes


if __name__ == "__main__":
    base = run("base_resumes.json", "pairs.json", SEED)

    # neutral version of each base resume, used for the accuracy metric (no real name, mid tier college)
    labelled = [{"id": r["id"], "difficulty": r["difficulty"], "label": r["label"],
                 "label_reason": r["label_reason"],
                 "text": fill(r["resume"], f"Candidate {r['id']}", NEUTRAL_COLLEGE)} for r in base]
    save("labelled_set.json", labelled)

    if (HERE / "unseen_resumes.json").exists():
        run("unseen_resumes.json", "unseen_pairs.json", SEED + 1, prefix="U")
    print("all checks passed")
