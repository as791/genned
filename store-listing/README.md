# Store listing (Uptodown, Google Play and other stores)

## Screenshots

`screenshots/`: six 1080 × 1920 PNGs, portrait, real app screens with a short caption.
Upload them in this order:

1. `genned-screenshot-01.png`: Is it AI? Ask Genned
2. `genned-screenshot-02.png`: A plain answer in seconds
3. `genned-screenshot-03.png`: Honest when it can't tell
4. `genned-screenshot-04.png`: Check Reels without leaving Instagram
5. `genned-screenshot-05.png`: Verifies Content Credentials
6. `genned-screenshot-06.png`: Private. Nothing is uploaded

`raw/`: the same app screens without captions (720 × 1561 JPEG), for stores that want
plain screenshots. Their aspect ratio is taller than 2:1, which Google Play does not accept;
use `screenshots/` there.

The screens come from the app's screenshot tests (v0.5.0) and show sample results.

## Featured image

`featured/`: the store banner. `genned-featured-1024x500.png` is Google Play's feature
graphic size; `genned-featured-1920x1080.png` is for stores that ask for a 16:9 image.

## Icon

`../site/assets/brand/`: `icon-square-512.png` for Google Play (512 × 512; the store
rounds the corners), `icon-rounded-512.png` for stores that show the icon as uploaded.

## Text

**App name:** Genned

**Short description (80 characters max):**
Check if a photo or video is AI-generated. Private, on your phone, free.

**Full description:**

Is that photo real? Is that Reel made by AI? Genned gives you a second opinion before you
believe it, forward it or reply.

Share any photo or video to Genned from WhatsApp, Instagram, your gallery or any app with
a Share button, and get a plain answer in seconds:
• Likely AI-generated
• Uncertain, and why
• Likely not AI-generated

Genned only says "likely AI-generated" at 90% or more, and "likely not" below 15%. In
between it tells you it can't tell, instead of guessing.

What it checks:
• Two AI detectors look at the picture itself.
• What the file says about itself: camera details and AI tool tags.
• Content Credentials: signed records from cameras, editors and AI tools such as ChatGPT,
  verified offline against the official C2PA trust lists.

Made for how you scroll:
• A bubble for Instagram and WhatsApp: tap it to check what's on your screen.
• Photos, screenshots, and videos such as Reels and Shorts up to 3 minutes.
• History on your phone, with small thumbnails only.

Private by design:
• Every check runs on your phone. Photos and videos are never uploaded.
• No account, no ads, no analytics.
• The app has no internet permission at all.

Free and open source (Apache-2.0).

Genned is an estimate, not proof. The newest AI tools and heavily edited images can fool
it, and it says "Uncertain" when it can't tell.

**Category:** Tools (or Photography)

**Website:** https://as791.github.io/genned/

**Privacy policy:** https://as791.github.io/genned/privacy.html

**Source code:** https://github.com/as791/genned

**Contact:** https://github.com/as791/genned/issues
