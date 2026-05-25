# ADetailer Neo
The good ol' [ADetailer](https://github.com/Bing-su/adetailer) extension, which performs automatic masking and inpainting, completely rewritten for [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo), now ***lighter, faster, and prettier*** :tm:

- **Download ZIP:** `68 KB` &rarr; `37 KB`

## Features
After a generation, use **YOLO** or **MediaPipe** model(s) to detect certain regions *(**e.g.** face)* to perform **Inpainting**, improving fine details automatically

## Models
On a fresh launch, this Extension will download the following **10** detector models into the `models/adetailer/` folder

<table>
	<tr>
		<th>Filename</th>
		<th>Source</th>
	</tr>
	<tr>
		<td>face_yolov8n.pt</td>
		<td rowspan="6">
			<a href="https://huggingface.co/Bingsu/adetailer">Bingsu/adetailer</a>
		</td>
	</tr>
	<tr>
		<td>face_yolov8s.pt</td>
	</tr>
	<tr>
		<td>hand_yolov8n.pt</td>
	</tr>
	<tr>
		<td>hand_yolov8s.pt</td>
	</tr>
	<tr>
		<td>person_yolov8n-seg.pt</td>
	</tr>
	<tr>
		<td>person_yolov8s-seg.pt</td>
	</tr>
	<tr>
		<td>yolov8x-worldv2.pt</td>
		<td>
			<a href="https://github.com/ultralytics/assets">ultralytics/assets</a>
		</td>
	</tr>
	<tr>
		<td>mediapipe_face_short.tflite</td>
		<td rowspan="2">
			<a href="https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector">mediapipe/face_detector</a>
		</td>
	</tr>
	<tr>
		<td>mediapipe_face_full.tflite</td>
	</tr>
	<tr>
		<td>face_landmarker.task</td>
		<td>
			<a href="https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker">mediapipe/face_landmarker</a>
		</td>
	</tr>
</table>

> [!Tip]
> You can put additional [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO models into the `models/adetailer/` folder

> [!Important]
> The **YOLO** models must be in `.pt` format

<hr>

<pre align="center">
Copyright (C) 2026 Bing-su
Copyright (C) 2026 Haoming02

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see https://www.gnu.org/licenses/.
</pre>
