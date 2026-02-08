import { useEffect, useRef, useState } from 'react';
import { fetchCameraPlayback } from '../api';
import { ENABLE_WHEP, MEDIAMTX_WHEP_BASE_URL, MEDIAMTX_WHEP_PATH_TEMPLATE } from '../config';

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

  const buildWhepUrl = (id: number) => {
    const path = MEDIAMTX_WHEP_PATH_TEMPLATE
      .replace('{camera_id}', String(id))
      .replace('{cameraId}', String(id));
    return `${MEDIAMTX_WHEP_BASE_URL}/${path.replace(/^\/+/, '').replace(/\/+$/, '')}/whep`;
  };

  useEffect(() => {
    let cancelled = false;

    if (!ENABLE_WHEP) {
      setState('idle');
      return undefined;
    }

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

      const playback = await fetchCameraPlayback(cameraId);
      const whepUrl = playback.whep_url || buildWhepUrl(cameraId);

      if (!peer.localDescription) {
        setState('failed');
        return;
      }

      if (peer.iceGatheringState !== 'complete') {
        await new Promise<void>((resolve) => {
          const handleStateChange = () => {
            if (peer.iceGatheringState === 'complete') {
              peer.removeEventListener('icegatheringstatechange', handleStateChange);
              resolve();
            }
          };
          peer.addEventListener('icegatheringstatechange', handleStateChange);
        });
      }

      const response = await fetch(whepUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/sdp',
          Accept: 'application/sdp'
        },
        body: peer.localDescription.sdp
      });

      if (!response.ok) {
        setState('failed');
        return;
      }

      const answerSdp = await response.text();
      await peer.setRemoteDescription(
        new RTCSessionDescription({
          type: 'answer',
          sdp: answerSdp
        })
      );
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
