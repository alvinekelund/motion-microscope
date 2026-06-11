export async function startCamera(): Promise<HTMLVideoElement> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Camera access (getUserMedia) is not available in this browser.");
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 30 },
      facingMode: "user",
    },
    audio: false,
  });
  const video = document.createElement("video");
  video.srcObject = stream;
  video.playsInline = true;
  video.muted = true;
  await video.play();
  if (!video.videoWidth) {
    await new Promise<void>((resolve) => {
      video.addEventListener("loadedmetadata", () => resolve(), { once: true });
    });
  }
  return video;
}
