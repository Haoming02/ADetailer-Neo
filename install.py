import launch

if not launch.is_installed("ultralytics"):
    launch.run_pip("install ultralytics==8.3.253", "ultralytics")

if not launch.is_installed("mediapipe"):
    launch.run_pip("install mediapipe==0.10.35", "mediapipe")
