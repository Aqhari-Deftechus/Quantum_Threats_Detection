import { useEffect, useRef, useState } from 'react';

type WebRTCPlayerProps = {
  cameraId: number;
  fallbackSrc: string;
  className?: string;
  onFrameDimensions?: (width: number, height: number) => void;
};

type PlayerState = 'idle' | 'connecting' | 'connected' | 'failed';

export default function WebRTCPlayer({
  cameraId,
  fallbackSrc,
  className,
  onFrameDimensions
}: WebRTCPlayerProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const peerRef = useRef<RTCPeerConnection | null>(null);
  const [state, setState] = useState<PlayerState>('idle');

  useEffect(() => {
    let cancelled = false;

    const connect = async () => {
      setState('connecting');
      const peer = new RTCPeerConnection();
      peerRef.current = peer;

      peer.ontrack = (event) => {
        if (videoRef.current) {
          const [stream] = event.streams;
          videoRef.current.srcObject = stream;
        }
      };

      peer.onconnectionstatechange = () => {
        if (cancelled) return;
        if (peer.connectionState === 'connected') {
          setState('connected');
        }
        if (peer.connectionState === 'failed' || peer.connectionState === 'disconnected') {
          setState('failed');
        }
      };

      peer.addTransceiver('video', { direction: 'recvonly' });

      const offer = await peer.createOffer();
      await peer.setLocalDescription(offer);

      const response = await fetch('/api/webrtc/offer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          camera_id: cameraId,
          sdp: offer.sdp,
          type: offer.type
        })
      });

      if (!response.ok) {
        setState('failed');
        return;
      }

      const data = (await response.json()) as { sdp: string; type: RTCSdpType };
      await peer.setRemoteDescription(new RTCSessionDescription(data));
      if (!cancelled) {
        setState('connected');
      }
    };

    connect().catch(() => {
      if (!cancelled) {
        setState('failed');
      }
    });

    return () => {
      cancelled = true;
      if (peerRef.current) {
        peerRef.current.close();
        peerRef.current = null;
      }
    };
  }, [cameraId]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return undefined;

    const handleLoaded = () => {
      if (onFrameDimensions && video.videoWidth && video.videoHeight) {
        onFrameDimensions(video.videoWidth, video.videoHeight);
      }
    };
    video.addEventListener('loadedmetadata', handleLoaded);
    return () => video.removeEventListener('loadedmetadata', handleLoaded);
  }, [onFrameDimensions, state]);

  if (state === 'connected') {
    return <video ref={videoRef} className={className} autoPlay playsInline muted />;
  }

  return <img className={className} src={fallbackSrc} alt="Camera stream" />;
}
