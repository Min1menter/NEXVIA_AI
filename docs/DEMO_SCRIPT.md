# Demo recording plan (about 5 minutes)

> FOR EDUCATION ONLY - NOT FOR CLINICAL USE.

## Tool
**OBS Studio** (free, obsproject.com). Sources: *Display Capture* + *Audio Input Capture* (your microphone).
Settings: Output > Recording Format **mp4**; Video 1920x1080, 30 fps; Output > Recording > bitrate about **3000 Kbps** (keeps 5 minutes near 100 MB).
Do a 20-second test recording first and listen to it. (Windows Game Bar records only one window, so it cannot follow you from browser to terminal.)

## Before you press record
- [ ] `docker compose ps` shows all five services running; `python scripts\smoke_test.py` passes 18/18.
- [ ] Browser: new window, **hide the bookmarks bar** (Ctrl+Shift+B), close other tabs, zoom 110%, open `http://localhost:8080` (not signed in).
- [ ] A second browser tab with `http://localhost:8042` (Orthanc) already logged in, and a third with `docs/architecture.png`.
- [ ] PowerShell in the repo folder, font size large, screen cleared (`cls`). **Never print `.env` or show passwords.**
- [ ] Notifications off (Windows Focus mode), phone silent, other apps closed.
- [ ] Pick your studies in advance: one chest X-ray with a clear overlay (e.g. the Pneumothorax or Atelectasis study), one CT, one pending study to approve, another to reject.
- [ ] Do the whole script once without recording.

## Script (talk while you click)
| Time | On screen | Say (in your own words) |
|---|---|---|
| 0:00 | Login page with the yellow banner | What the project is; it is education only, not clinical use. |
| 0:20 | `architecture.png` | One sentence per box: Orthanc stores DICOM, the worker runs the models, the API and viewer let a radiologist review, everything is audited. |
| 0:45 | Signed in, worklist | 50 studies, chest X-ray and CT; each shows the top finding, a score and a status. |
| 1:05 | Orthanc tab: studies list, then one study | The same studies live in the DICOM server; the AI result is written back as an extra series. |
| 1:25 | A chest X-ray: overlay on/off, opacity, finding dropdown | Grad-CAM shows which region influenced the prediction; top-3 findings have overlays; the score is uncalibrated and the heat map is coarse. |
| 2:10 | A CT study | 3D U-Net spleen mask, volume in mL, slice shown. |
| 2:35 | Approve one study; reject another (type the reason first, then confirm) | Only a radiologist can decide; rejecting needs a reason; each decision is final. |
| 3:15 | Audit log tab, then Verify hash chain | Every login, view and decision is logged; each row's hash covers the previous row. |
| 3:40 | Terminal: tamper demo (commands below), then Verify again | The database refuses edits; even with the protection removed, the hash chain detects the change. |
| 4:30 | Model card or results slide | Classifier: validation AUROC 0.80; 0.84 over classes with enough positives; spleen Dice 0.92. Honest limits: one class at chance, several unscoreable, scores uncalibrated. |
| 4:55 | Back to the banner | Education only. Thanks. |

## Terminal commands for the tamper demo (paste one at a time)
```powershell
docker compose exec -T db psql -U medai -d medai -c "UPDATE audit_log SET actor='mallory' WHERE id=2;"
$orig = (docker compose exec -T db psql -U medai -d medai -t -A -c "SELECT actor FROM audit_log WHERE id=2;").Trim()
docker compose exec -T db psql -U medai -d medai -c "ALTER TABLE audit_log DISABLE TRIGGER USER;"
docker compose exec -T db psql -U medai -d medai -c "UPDATE audit_log SET actor='mallory' WHERE id=2;"
# -> click "Verify hash chain" in the viewer: tampering detected at entry 2
docker compose exec -T db psql -U medai -d medai -c "UPDATE audit_log SET actor='$orig' WHERE id=2;"
docker compose exec -T db psql -U medai -d medai -c "ALTER TABLE audit_log ENABLE TRIGGER USER;"
# -> click "Verify hash chain" again: intact
```
The first command should fail with "audit_log is append-only". **If anything goes wrong, always finish with the last two commands** so the database is restored.

## After recording
- Watch it once at full screen: audio audible, no passwords or personal tabs visible, text readable.
- Optional trim: Clipchamp (built into Windows) or just re-record the bad section.
- Save as `demo.mp4`. **Do not commit it to Git** (large); upload it to the Drive folder (or YouTube as *unlisted*) and put the link in the README.
- While the stack is still running, also retake: `01_compose_ps` with
  `docker compose ps --format "table {{.Service}}\t{{.Status}}\t{{.Ports}}"`, `09_rejected_reason` (reason typed in), and `16_audit_restored_ok`.
