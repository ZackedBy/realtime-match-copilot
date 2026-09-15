# Model Release Cycle

## Versioning

- Format: `major.minor.patch` for recommendation/vision bundles.
- Every bundle stores metrics snapshot and dataset tag.

## Promotion gates

1. Offline replay evaluation passes target scenarios.
2. Shadow run in live matches with no user-facing cards.
3. Safety gate checks:
   - min FPS
   - fail-safe ratio ceiling
   - recommendation quality proxy floor

## Rollback

- Automatic rollback when drift detector is triggered repeatedly in production windows.
- Keep last known-good bundle as default fallback.

