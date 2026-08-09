<!-- Modified by NextoCR from the Apache-2.0 crforge pull request template. -->
## Summary
<!-- Brief description of the changes -->

## Changes
<!-- Bullet list of what changed -->
-

## Card/Mechanic affected
<!-- If applicable, which card or game mechanic is affected? -->

## Data provenance
<!-- For gameplay-data changes: source URL, revision/patch date, level, confidence, and method. -->

## Test plan
<!-- How were the changes verified? -->
- [ ] Ran `./gradlew clean build`
- [ ] Ran `python scripts/card_catalog_report.py --strict`
- [ ] Manual testing in debug visualizer

## Checklist
- [ ] Tests pass (`./gradlew clean build`)
- [ ] Ran `./gradlew spotlessApply`
- [ ] Determinism/concurrency changes include multi-session reset/replay coverage
- [ ] Card-data changes include allowed source, version, attribution, and validation
- [ ] No official art/audio/client assets or live-client automation added
- [ ] No secrets or credentials committed
