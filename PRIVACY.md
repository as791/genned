# Genned privacy policy

_Effective 1 October 2026. Applies to the Genned Android app, as distributed from
[GitHub Releases](https://github.com/as791/genned/releases)._

**In short:** Genned checks images and videos entirely on your phone. It has no servers,
no accounts, no ads and no analytics. It doesn't have Android's internet permission, so it
cannot send anything anywhere.

## What Genned collects

Nothing. The developer receives no data from the app: no images, videos, results, usage
statistics, crash reports, identifiers or location.

## What happens to an image or video you check

- You share it into Genned or pick it with Android's photo picker. Android gives Genned
  temporary access to that one file only. Genned never asks for access to your photo
  library or storage.
- A working copy is made in the app's private storage and analyzed on the phone: an AI
  image classifier, plus the file's own metadata (EXIF, generator tags, provenance
  data).
- To tell whether a shared image is a screenshot, Genned looks at the file's name and
  compares the image's size with your screen, on the phone. If it is one, only the picture
  inside it is checked. The file name is not stored or logged.
- The working copy is deleted when the check finishes. Leftovers from a failed or
  cancelled check are deleted the next time the app starts.

## What stays on your phone

- **History:** for each check, the result (score, classification, the reasons
  behind it) and a small thumbnail (at most 256 px). The full image is never kept.
- History is stored in the app's private storage. Other apps can't read it, and it is
  excluded from Android's cloud backup.
- You can delete one entry or clear all of history in the app. Uninstalling the app
  deletes everything.

## Sharing a result

When you tap "Share result", Genned draws a card from the score and reasons only. Your
image is never included. The card goes only to the app you choose in Android's share
sheet. What happens after that is governed by that app's privacy policy.

## Screen overlay (optional, off by default)

Settings → Experimental → "Enable overlay bubble" lets you check what's on screen in
Instagram or WhatsApp. It only runs if you turn it on and grant three Android
permissions:

| Permission | Why |
|---|---|
| Display over other apps | To draw the floating bubble. |
| Usage access | To know which app is in front, so the bubble shows only in Instagram or WhatsApp. It reveals app names only, never what's in them. |
| Screen capture (asked for every time the overlay starts) | Gives the app a live copy of your screen while Instagram or WhatsApp is open. Genned reads one frame from it only when you tap the bubble. |

- While the overlay is on, Android shows a notification you can't hide, with a Stop
  button.
- A frame you capture is handled exactly like a shared image: analyzed on the phone,
  then deleted, with only the result and a thumbnail kept in history.
- Genned never reads other apps' data, messages, network traffic or accessibility
  information.
- Turning the overlay off, or tapping Stop, ends screen capture immediately.

## Permissions

Genned asks only for the overlay permissions above, and only when you turn the overlay
on. It has no internet, camera, microphone, location, contacts or storage permission.

## Children

Genned is not directed at children and collects no data from anyone.

## Changes

If this policy changes, the new version will be published here and noted in the
release that introduces the change. A version of Genned that sent any data off your
phone would need the internet permission. That would be stated in this policy and in
the release notes before it shipped.

## Contact

Questions or concerns: open an issue at https://github.com/as791/genned/issues.
