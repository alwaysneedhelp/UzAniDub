# UzAniDub website

A library for Uzbek dubbed anime: anyone can browse, watch, add anime and
upload episodes, with no account. Catalog metadata lives in Cloudflare
D1; covers, episode videos, and episode thumbnails live in R2.

Pages:

- `/` — featured carousel, Continue Watching, Most Popular, New Episodes, and genre rows
- `/browse` — every anime, sorted by popularity, newest, or A–Z, with genre filters
- `/search?q=` — search by title, description, or genre
- `/series/[id]` — anime details and its episode list
- `/watch/[id]` — the player, with resume, view counting, and an "Up next" countdown
- `/upload` — add an anime (with cover) or upload a dubbed episode

Episode thumbnails and durations are captured in the uploader's browser before
the video upload starts. Watch progress is stored in each viewer's browser.

## Local setup

```bash
cd site
npm ci
npx wrangler d1 execute DB --local --config wrangler.local.jsonc --persist-to .wrangler/state --file drizzle/0000_cute_greymalkin.sql
npx wrangler d1 execute DB --local --config wrangler.local.jsonc --persist-to .wrangler/state --file drizzle/0001_pretty_shape.sql
npx wrangler d1 execute DB --local --config wrangler.local.jsonc --persist-to .wrangler/state --file drizzle/0002_episode_media.sql
npm run dev
```

Apply each migration once, then open `http://localhost:3000`.

Uploading is open: nobody needs to sign in to add an anime or upload an
episode. Anyone who can reach the site can publish to it, so keep hosted access
private until distribution rights for uploaded media and art are confirmed, or
add an uploader check in `lib/auth.ts` before opening it up.

The dubbing engine (`src/dubtool`, started with `run_engine.py`) is separate
from the website; publish its finished videos through the upload page.

Run `npm test` for TypeScript checks and `npm run build` for the deployment
build. Sites applies Drizzle migrations on publishing.
