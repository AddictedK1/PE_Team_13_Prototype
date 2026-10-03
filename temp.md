@"
# Temporary Notes

This file is for temporary development notes and testing.

## TODO

- Review prototype
- Run evaluation
- Verify counterfactual pairs
- Test unseen candidate
- Verify final demo
"@ | Set-Content -Encoding UTF8 temp.md

git status
git add .\temp.md
git commit -m "docs: add temporary project notes"
git push
