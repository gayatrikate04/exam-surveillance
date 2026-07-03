"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  checkDetectionApiHealth,
  deleteLiveSession,
  getApiBaseUrl,
  isUnreachableApiError,
  processImageApi,
  processLiveFrameApi,
  formatDetectionApiError,
  processVideoApi,
  type ImageProcessResponse,
  type LiveFrameResponse,
  type Summary,
  type VideoProcessResponse,
} from "@/lib/api";
import { postClientAudit } from "@/lib/client-audit";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

type InputMode = "image" | "video" | "live";

/** Stable Select value — never collides with a real `MediaDeviceInfo.deviceId`. */
const DEFAULT_CAMERA_SELECT_VALUE = "__browser_default__";

function imageDataUrl(b64: string, mime = "image/png") {
  return `data:${mime};base64,${b64}`;
}

function sliderFirst(v: number | readonly number[]): number {
  return Array.isArray(v) ? Number(v[0]) : Number(v);
}

function triggerDownload(dataUrl: string, filename: string) {
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = filename;
  a.click();
}

function Dashboard({
  summary,
  compact,
}: {
  summary: Summary | null;
  compact: boolean;
}) {
  if (!summary) return null;
  const people = Number(summary.people ?? 0);
  const hasObjects = summary.objects && Object.keys(summary.objects).length > 0;
  if (people === 0 && !hasObjects) return null;

  if (compact) {
    const risk = String(summary.overall_risk ?? "low").toUpperCase();
    return (
      <div className="space-y-2 text-sm text-muted-foreground">
        <p>
          <strong>{people}</strong> people · <strong>{summary.normal ?? 0}</strong> ok ·{" "}
          <strong>{summary.suspicious ?? 0}</strong> flagged · risk <strong>{risk}</strong>
        </p>
        {Number(summary.back_turned ?? 0) > 0 && (
          <Alert variant="destructive">
            <AlertTitle>Back-facing</AlertTitle>
            <AlertDescription>
              Detected for <strong>{String(summary.back_turned)}</strong> person(s).
            </AlertDescription>
          </Alert>
        )}
        {Number(summary.phone_near_person ?? 0) > 0 && (
          <Alert>
            <AlertTitle>Phone</AlertTitle>
            <AlertDescription>
              {String(summary.phone_near_person)} student(s) near a detected phone.
            </AlertDescription>
          </Alert>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Card>
          <CardHeader className="p-3 pb-1">
            <CardDescription>People</CardDescription>
            <CardTitle className="text-2xl">{people}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="p-3 pb-1">
            <CardDescription>Normal</CardDescription>
            <CardTitle className="text-2xl">{summary.normal ?? 0}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="p-3 pb-1">
            <CardDescription>Suspicious</CardDescription>
            <CardTitle className="text-2xl">{summary.suspicious ?? 0}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="p-3 pb-1">
            <CardDescription>Risk</CardDescription>
            <CardTitle className="text-2xl uppercase">{String(summary.overall_risk ?? "low")}</CardTitle>
          </CardHeader>
        </Card>
      </div>
      {Number(summary.back_turned ?? 0) > 0 && (
        <Alert variant="destructive">
          <AlertTitle>Back-facing</AlertTitle>
          <AlertDescription>
            {String(summary.back_turned)} person(s) appear to be looking back.
          </AlertDescription>
        </Alert>
      )}
      {Number(summary.phone_near_person ?? 0) > 0 && (
        <Alert>
          <AlertTitle>Phone near student</AlertTitle>
          <AlertDescription>{String(summary.phone_near_person)} association(s).</AlertDescription>
        </Alert>
      )}
      {hasObjects && (
        <p className="text-xs text-muted-foreground">
          Objects:{" "}
          {Object.entries(summary.objects as Record<string, number>)
            .map(([k, v]) => `${k}: ${v}`)
            .join(", ")}
        </p>
      )}
    </div>
  );
}

export function ExamMonitor() {
  const [minimalUi, setMinimalUi] = useState(true);
  const [inputMode, setInputMode] = useState<InputMode>("image");

  const [multiPerson, setMultiPerson] = useState(true);
  const [crowd, setCrowd] = useState(true);
  const [maxPeople, setMaxPeople] = useState(200);
  const [yoloConf, setYoloConf] = useState(0.12);
  const [yoloImgsz, setYoloImgsz] = useState(1280);
  const [yoloIou, setYoloIou] = useState(0.55);
  const [kptFloor, setKptFloor] = useState(0.12);
  const [objectHires, setObjectHires] = useState(true);
  const [objectConf, setObjectConf] = useState(0.18);
  const [objectTta, setObjectTta] = useState(false);

  /** No top status strip on the annotated frame (`none` in detector); use `full` in Advanced for debug banner. */
  const examOverlay = minimalUi ? "none" : "full";

  const inferenceParams = useMemo(() => {
    const yConf = minimalUi ? 0.12 : crowd ? 0.12 : 0.22;
    const yImg = minimalUi ? 1280 : crowd ? 1280 : 960;
    const yIou = minimalUi ? 0.55 : crowd ? 0.55 : 0.6;
    const kf = minimalUi ? 0.12 : crowd ? 0.12 : 0.18;
    const oc = minimalUi ? 0.18 : crowd ? 0.18 : 0.22;
    return {
      multi_person: minimalUi ? true : multiPerson,
      yolo_conf: minimalUi ? yConf : yoloConf,
      yolo_imgsz: minimalUi ? yImg : yoloImgsz,
      max_people: minimalUi ? 200 : maxPeople,
      yolo_iou: minimalUi ? yIou : yoloIou,
      kpt_floor: minimalUi ? kf : kptFloor,
      object_detect_conf: minimalUi ? oc : objectConf,
      object_detect_hires: minimalUi ? true : objectHires,
      object_tta: minimalUi ? false : objectTta,
      exam_overlay: examOverlay,
    };
  }, [
    minimalUi,
    crowd,
    multiPerson,
    yoloConf,
    yoloImgsz,
    maxPeople,
    yoloIou,
    kptFloor,
    objectConf,
    objectHires,
    objectTta,
    examOverlay,
  ]);

  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageResult, setImageResult] = useState<ImageProcessResponse | null>(null);
  const [imageBusy, setImageBusy] = useState(false);
  const [imageError, setImageError] = useState<string | null>(null);

  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoResult, setVideoResult] = useState<VideoProcessResponse | null>(null);
  const [videoBusy, setVideoBusy] = useState(false);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [videoProgress, setVideoProgress] = useState(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const liveSessionRef = useRef<string>("");
  const [liveOn, setLiveOn] = useState(false);
  const [liveFrame, setLiveFrame] = useState<LiveFrameResponse | null>(null);
  const [liveError, setLiveError] = useState<string | null>(null);
  const prevBackAlert = useRef(false);
  const [liveFlash, setLiveFlash] = useState<string | null>(null);
  const liveApiBackoffUntilRef = useRef(0);
  const liveFetchFailuresRef = useRef(0);
  /** Throttle + signature for `app.live.status` audit rows (admin dashboard graph). */
  const lastLiveAuditAtRef = useRef(0);
  const lastLiveSigRef = useRef("");
  const prevRiskForAuditRef = useRef("low");

  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>(DEFAULT_CAMERA_SELECT_VALUE);
  const [deviceListHint, setDeviceListHint] = useState<string | null>(null);

  const selectedCameraLabel = useMemo(() => {
    if (selectedDeviceId === DEFAULT_CAMERA_SELECT_VALUE) return "Browser default";
    const d = videoDevices.find((x) => x.deviceId === selectedDeviceId);
    if (!d) return selectedDeviceId ? "Unknown device" : "Browser default";
    if (d.label && d.label.trim()) return d.label.trim();
    return `Camera (${d.deviceId.slice(0, 8)}…)`;
  }, [videoDevices, selectedDeviceId]);

  const refreshVideoDevices = useCallback(async () => {
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.enumerateDevices) {
      setDeviceListHint("This browser does not support listing cameras.");
      return;
    }
    try {
      const list = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
      setVideoDevices(list);
      setSelectedDeviceId((cur) => {
        if (cur === DEFAULT_CAMERA_SELECT_VALUE) return DEFAULT_CAMERA_SELECT_VALUE;
        if (list.some((d) => d.deviceId === cur)) return cur;
        return list[0]?.deviceId ?? DEFAULT_CAMERA_SELECT_VALUE;
      });
      const unnamed = list.filter((d) => !d.label).length;
      setDeviceListHint(
        unnamed > 0
          ? "Some cameras show generic names until you allow access once; click Start or Refresh again."
          : null,
      );
    } catch {
      setDeviceListHint("Could not enumerate cameras.");
    }
  }, []);

  useEffect(() => {
    if (inputMode === "live") void refreshVideoDevices();
  }, [inputMode, refreshVideoDevices]);

  const runImage = async () => {
    if (!imageFile) return;
    setImageBusy(true);
    setImageError(null);
    try {
      const res = await processImageApi(imageFile, inferenceParams as Record<string, string | number | boolean>);
      setImageResult(res);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Image processing failed";
      setImageError(msg);
      postClientAudit({
        level: "warn",
        action: "app.process.image_failed",
        message: msg,
      });
    } finally {
      setImageBusy(false);
    }
  };

  const runVideo = async () => {
    if (!videoFile) return;
    setVideoBusy(true);
    setVideoError(null);
    setVideoProgress(5);
    try {
      const params = {
        ...inferenceParams,
        yolo_imgsz: Math.min(Number(inferenceParams.yolo_imgsz), 960),
      };
      const res = await processVideoApi(videoFile, params as Record<string, string | number | boolean>);
      setVideoResult(res);
      setVideoProgress(100);
    } catch (e) {
      const msg = formatDetectionApiError(e, { uploadBytes: videoFile.size });
      setVideoError(msg);
      postClientAudit({
        level: "warn",
        action: "app.process.video_failed",
        message: msg,
      });
    } finally {
      setVideoBusy(false);
    }
  };

  const stopLive = useCallback(async () => {
    liveApiBackoffUntilRef.current = 0;
    liveFetchFailuresRef.current = 0;
    setLiveOn(false);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (liveSessionRef.current) {
      const sid = liveSessionRef.current;
      await deleteLiveSession(sid);
      postClientAudit({
        level: "info",
        action: "app.live.stop",
        message: "Live monitoring stopped",
        meta: { sessionId: sid },
      });
      liveSessionRef.current = "";
    }
    lastLiveAuditAtRef.current = 0;
    lastLiveSigRef.current = "";
    prevRiskForAuditRef.current = "low";
    prevBackAlert.current = false;
    setLiveFlash(null);
  }, []);

  const startLive = async () => {
    setLiveError(null);
    liveApiBackoffUntilRef.current = 0;
    liveFetchFailuresRef.current = 0;
    const base = getApiBaseUrl();
    const health = await checkDetectionApiHealth();
    if (!health.ok) {
      setLiveError(
        `Detection API is not reachable at ${base} (${health.detail}). ` +
          `From the project folder run run_web_stack.bat (starts API + web), or start only the API: ` +
          `uvicorn api_server:app --host 127.0.0.1 --port 8000`,
      );
      return;
    }
    try {
      const video: MediaTrackConstraints = {
        width: { ideal: 1280 },
        height: { ideal: 720 },
      };
      if (
        selectedDeviceId !== DEFAULT_CAMERA_SELECT_VALUE &&
        videoDevices.some((d) => d.deviceId === selectedDeviceId)
      ) {
        video.deviceId = { exact: selectedDeviceId };
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        video,
        audio: false,
      });
      streamRef.current = stream;
      const track = stream.getVideoTracks()[0];
      const settings = track?.getSettings?.();
      if (settings?.deviceId) {
        setSelectedDeviceId(settings.deviceId);
      }
      await refreshVideoDevices();
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      liveSessionRef.current = crypto.randomUUID();
      prevBackAlert.current = false;
      lastLiveAuditAtRef.current = 0;
      lastLiveSigRef.current = "";
      prevRiskForAuditRef.current = "low";
      setLiveOn(true);
      postClientAudit({
        level: "info",
        action: "app.live.start",
        message: "Live webcam monitoring started",
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Could not open webcam";
      setLiveError(msg);
      postClientAudit({
        level: "warn",
        action: "app.live.webcam_error",
        message: msg,
      });
    }
  };

  useEffect(() => {
    if (!liveOn) return;
    let cancelled = false;
    const tick = async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;
      const w = video.videoWidth;
      const h = video.videoHeight;
      if (!w || !h) return;
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, w, h);
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob((b) => resolve(b), "image/jpeg", 0.85),
      );
      if (!blob || cancelled) return;
      if (Date.now() < liveApiBackoffUntilRef.current) return;
      try {
        const params = {
          ...inferenceParams,
          yolo_imgsz: Math.min(Number(inferenceParams.yolo_imgsz), 960),
        };
        const res = await processLiveFrameApi(
          blob,
          liveSessionRef.current,
          params as Record<string, string | number | boolean>,
        );
        if (cancelled) return;
        liveFetchFailuresRef.current = 0;
        liveApiBackoffUntilRef.current = 0;
        setLiveError((prev) => (prev !== null ? null : prev));
        liveSessionRef.current = res.session_id;
        setLiveFrame(res);
        const summary = res.summary;
        const sig = `${String(summary.overall_risk ?? "")}|${Number(summary.people ?? 0)}|${Number(summary.suspicious ?? 0)}|${Number(summary.back_turned ?? 0)}|${Number(summary.phone_near_person ?? 0)}`;
        const nowMs = Date.now();
        if (sig !== lastLiveSigRef.current || nowMs - lastLiveAuditAtRef.current >= 6000) {
          lastLiveSigRef.current = sig;
          lastLiveAuditAtRef.current = nowMs;
          postClientAudit({
            level: "info",
            action: "app.live.status",
            message: `Live: ${Number(summary.people ?? 0)} people · ${String(summary.overall_risk ?? "?").toUpperCase()} risk`,
            meta: {
              people: summary.people,
              normal: summary.normal,
              suspicious: summary.suspicious,
              overall_risk: summary.overall_risk,
              back_turned: summary.back_turned,
              phone_near_person: summary.phone_near_person,
            },
          });
        }
        const risk = String(summary.overall_risk ?? "low").toLowerCase();
        const prevRisk = prevRiskForAuditRef.current;
        if ((risk === "medium" || risk === "high") && prevRisk === "low") {
          postClientAudit({
            level: "warn",
            action: "app.live.risk_alert",
            message: `Live risk elevated to ${risk.toUpperCase()}`,
            meta: {
              overall_risk: risk,
              people: summary.people,
              suspicious: summary.suspicious,
            },
          });
        } else if (risk === "high" && prevRisk === "medium") {
          postClientAudit({
            level: "warn",
            action: "app.live.risk_alert",
            message: "Live risk elevated to HIGH",
            meta: {
              overall_risk: risk,
              people: summary.people,
              suspicious: summary.suspicious,
            },
          });
        }
        prevRiskForAuditRef.current = risk;
        const back = Number(res.summary.back_turned ?? 0);
        const on = back > 0;
        if (on && !prevBackAlert.current) {
          const msg = `ALERT: ${back} person(s) appear to be looking back.`;
          setLiveFlash(msg);
          postClientAudit({
            level: "warn",
            action: "app.detection.back_turned",
            message: msg,
            meta: { count: back },
          });
          try {
            if (typeof window !== "undefined" && "Notification" in window) {
              if (Notification.permission === "default") {
                void Notification.requestPermission();
              }
              if (Notification.permission === "granted") {
                new Notification("Exam monitor", { body: msg });
              }
            }
          } catch {
            /* ignore */
          }
        }
        prevBackAlert.current = on;
        if (!on) setLiveFlash(null);
      } catch (e) {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : "Live frame failed";
        if (isUnreachableApiError(e)) {
          liveFetchFailuresRef.current += 1;
          if (liveFetchFailuresRef.current === 1) {
            const base = getApiBaseUrl();
            postClientAudit({
              level: "warn",
              action: "app.live.api_unreachable",
              message: `Cannot reach detection API at ${base}`,
              meta: { detail: msg },
            });
          }
          const streak = Math.min(liveFetchFailuresRef.current, 6);
          const ms = Math.min(30_000, 500 * 2 ** streak);
          liveApiBackoffUntilRef.current = Date.now() + ms;
          const apiBase = getApiBaseUrl();
          const unreachableMsg =
            `Cannot reach the detection API at ${apiBase} (${msg}). ` +
            `From the project folder run run_web_stack.bat, or: uvicorn api_server:app --host 127.0.0.1 --port 8000`;
          setLiveError((prev) => (prev === unreachableMsg ? prev : unreachableMsg));
        } else {
          liveFetchFailuresRef.current = 0;
          liveApiBackoffUntilRef.current = 0;
          setLiveError(msg);
        }
      }
    };
    const id = window.setInterval(tick, 220);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [liveOn, inferenceParams]);

  useEffect(() => {
    return () => {
      void stopLive();
    };
  }, [stopLive]);

  useEffect(() => {
    if (inputMode !== "live" && liveOn) void stopLive();
  }, [inputMode, liveOn, stopLive]);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 lg:flex-row">
      <aside className="w-full shrink-0 space-y-4 border-b pb-4 lg:w-80 lg:border-b-0 lg:border-r lg:pr-4">
        <div>
          <h2 className="text-lg font-semibold">Controls</h2>
        </div>
        <div className="flex items-center justify-between gap-2">
          <Label>Interface</Label>
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">Minimal</span>
            <Switch checked={!minimalUi} onCheckedChange={(v) => setMinimalUi(!v)} />
            <span className="text-xs text-muted-foreground">Advanced</span>
          </div>
        </div>
        <Tabs value={inputMode} onValueChange={(v) => setInputMode(v as InputMode)}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="image">Image</TabsTrigger>
            <TabsTrigger value="video">Video</TabsTrigger>
            <TabsTrigger value="live">Live</TabsTrigger>
          </TabsList>
        </Tabs>

        {!minimalUi && (
          <>
            <div className="flex items-center justify-between">
              <Label>Multiple people</Label>
              <Switch checked={multiPerson} onCheckedChange={setMultiPerson} />
            </div>
            <div className="flex items-center justify-between">
              <Label>Boost distant</Label>
              <Switch checked={crowd} onCheckedChange={setCrowd} />
            </div>
            <div className="space-y-2">
              <Label>Max people: {maxPeople}</Label>
              <Slider
                value={[maxPeople]}
                min={10}
                max={250}
                step={10}
                onValueChange={(v) => setMaxPeople(sliderFirst(v))}
              />
            </div>
            <div className="space-y-2">
              <Label>Box confidence: {yoloConf.toFixed(2)}</Label>
              <Slider
                value={[yoloConf]}
                min={0.05}
                max={0.45}
                step={0.01}
                onValueChange={(v) => setYoloConf(sliderFirst(v))}
              />
            </div>
            <div className="space-y-2">
              <Label>Inference size</Label>
              <RadioGroup
                value={String(yoloImgsz)}
                onValueChange={(v) => setYoloImgsz(Number(v))}
                className="flex gap-3"
              >
                {[640, 960, 1280].map((sz) => (
                  <div key={sz} className="flex items-center gap-2">
                    <RadioGroupItem value={String(sz)} id={`sz-${sz}`} />
                    <Label htmlFor={`sz-${sz}`}>{sz}</Label>
                  </div>
                ))}
              </RadioGroup>
            </div>
            <div className="space-y-2">
              <Label>NMS IoU: {yoloIou.toFixed(2)}</Label>
              <Slider
                value={[yoloIou]}
                min={0.35}
                max={0.75}
                step={0.05}
                onValueChange={(v) => setYoloIou(sliderFirst(v))}
              />
            </div>
            <div className="space-y-2">
              <Label>Keypoint floor: {kptFloor.toFixed(2)}</Label>
              <Slider
                value={[kptFloor]}
                min={0.05}
                max={0.35}
                step={0.01}
                onValueChange={(v) => setKptFloor(sliderFirst(v))}
              />
            </div>
            <Separator />
            <div className="flex items-center justify-between">
              <Label>Higher-res object pass</Label>
              <Switch checked={objectHires} onCheckedChange={setObjectHires} />
            </div>
            <div className="space-y-2">
              <Label>Object confidence: {objectConf.toFixed(2)}</Label>
              <Slider
                value={[objectConf]}
                min={0.08}
                max={0.45}
                step={0.01}
                onValueChange={(v) => setObjectConf(sliderFirst(v))}
              />
            </div>
            <div className="flex items-center justify-between">
              <Label>Object TTA</Label>
              <Switch checked={objectTta} onCheckedChange={setObjectTta} />
            </div>
          </>
        )}

        {inputMode === "image" && (
          <div className="space-y-2">
            <Label>Image</Label>
            <Input type="file" accept="image/jpeg,image/png,image/jpg" onChange={(e) => setImageFile(e.target.files?.[0] ?? null)} />
            <Button disabled={!imageFile || imageBusy} onClick={runImage}>
              {imageBusy ? "Processing…" : "Run detection"}
            </Button>
          </div>
        )}

        {inputMode === "video" && (
          <div className="space-y-2">
            <Label>Video</Label>
            <Input type="file" accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska" onChange={(e) => setVideoFile(e.target.files?.[0] ?? null)} />
            <Button disabled={!videoFile || videoBusy} onClick={runVideo}>
              {videoBusy ? "Processing…" : "Sample 20 frames"}
            </Button>
            {videoBusy && <Progress value={videoProgress} />}
          </div>
        )}

        {inputMode === "live" && (
          <div className="space-y-3">
            <p className="text-xs text-muted-foreground">
              Browser webcam — frames are JPEG-encoded and sent to the API. Ensure{" "}
              <code className="rounded bg-muted px-1">uvicorn api_server:app --port 8000</code> is running.
            </p>
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-2">
                <Label className="shrink-0">Camera</Label>
                <Button type="button" variant="outline" size="sm" className="h-7 text-xs" onClick={() => void refreshVideoDevices()}>
                  Refresh list
                </Button>
              </div>
              <Select
                value={selectedDeviceId}
                onValueChange={(id: string | null) => {
                  if (id != null) setSelectedDeviceId(id);
                }}
                disabled={liveOn}
              >
                <SelectTrigger className="w-full min-w-0">
                  <SelectValue placeholder="Camera" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={DEFAULT_CAMERA_SELECT_VALUE}>Browser default</SelectItem>
                  {videoDevices.map((d) => (
                    <SelectItem key={d.deviceId} value={d.deviceId}>
                      {d.label && d.label.trim() ? d.label : `Camera (${d.deviceId.slice(0, 8)}…)`}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                <span className="font-medium text-foreground">Selected:</span> {selectedCameraLabel}
              </p>
              {deviceListHint && <p className="text-xs text-amber-700 dark:text-amber-400">{deviceListHint}</p>}
            </div>
            <div className="flex gap-2">
              <Button onClick={startLive} disabled={liveOn}>
                Start live
              </Button>
              <Button variant="secondary" onClick={() => void stopLive()} disabled={!liveOn}>
                Stop
              </Button>
            </div>
          </div>
        )}

        <Separator />
        <Collapsible>
          <CollapsibleTrigger className="flex w-full items-center justify-between text-sm font-medium">
            About
            <Badge variant="outline">info</Badge>
          </CollapsibleTrigger>
          <CollapsibleContent className="pt-2 text-xs text-muted-foreground">
            YOLOv8-Pose + objects. Suspicious posture on video/live uses temporal rules (scanning, back-facing). Photos are
            neutral except phone proximity. Tune in Advanced or via env vars on the API host.
          </CollapsibleContent>
        </Collapsible>
      </aside>

      <main className="min-h-0 flex-1 space-y-4 overflow-auto">
        {imageError && (
          <Alert variant="destructive">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{imageError}</AlertDescription>
          </Alert>
        )}
        {videoError && (
          <Alert variant="destructive">
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{videoError}</AlertDescription>
          </Alert>
        )}
        {liveError && (
          <Alert variant="destructive">
            <AlertTitle>Live</AlertTitle>
            <AlertDescription>{liveError}</AlertDescription>
          </Alert>
        )}
        {liveFlash && (
          <Alert variant="destructive">
            <AlertTitle>Alert</AlertTitle>
            <AlertDescription>{liveFlash}</AlertDescription>
          </Alert>
        )}

        {inputMode === "image" && imageResult && (
          <>
            <Dashboard summary={imageResult.summary} compact={minimalUi} />
            {!minimalUi && imageResult.summary.posture_mode === "snapshot_neutral" && (
              <p className="text-xs text-muted-foreground">
                Single image: posture not scored except phone near student. Use Video or Live for motion rules.
              </p>
            )}
            <div className={`grid gap-4 ${minimalUi ? "grid-cols-1" : "grid-cols-1 lg:grid-cols-2"}`}>
              {!minimalUi && (
                <Card>
                  <CardHeader>
                    <CardTitle>Original</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={imageDataUrl(imageResult.original_png_base64)} alt="Original" className="w-full rounded-md border" />
                  </CardContent>
                </Card>
              )}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between">
                  <CardTitle>Result</CardTitle>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() =>
                      triggerDownload(imageDataUrl(imageResult.annotated_png_base64), "exam_monitor_annotated.png")
                    }
                  >
                    Download PNG
                  </Button>
                </CardHeader>
                <CardContent>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={imageDataUrl(imageResult.annotated_png_base64)} alt="Annotated" className="w-full rounded-md border" />
                </CardContent>
              </Card>
            </div>
            {!minimalUi && imageResult.statuses.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Per person</CardTitle>
                </CardHeader>
                <CardContent className="space-y-1 text-sm">
                  {imageResult.statuses.map((s, i) => (
                    <div key={i}>
                      Person {i + 1}: {s.label} ({Math.round(s.confidence * 100)}%)
                    </div>
                  ))}
                </CardContent>
              </Card>
            )}
          </>
        )}

        {inputMode === "video" && videoResult && (
          <>
            <Dashboard summary={videoResult.last_summary} compact={minimalUi} />
            <ScrollArea className="h-[70vh] rounded-md border p-2">
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {videoResult.frames.map((f) => (
                  <Card key={f.index}>
                    <CardHeader className="py-2">
                      <CardTitle className="text-sm">Frame {f.index}</CardTitle>
                      <CardDescription>
                        N:{f.statuses.filter((x) => x.label === "Normal").length} S:
                        {f.statuses.filter((x) => x.label === "Suspicious").length}
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={imageDataUrl(f.annotated_png_base64, f.media_type ?? "image/png")}
                        alt=""
                        className="w-full rounded-md"
                      />
                    </CardContent>
                  </Card>
                ))}
              </div>
            </ScrollArea>
          </>
        )}

        {inputMode === "live" && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <Badge variant="secondary">Live</Badge>
              <span className="text-muted-foreground">Camera:</span>
              <span className="font-medium">{selectedCameraLabel}</span>
            </div>
            <video ref={videoRef} className="hidden" playsInline muted />
            <canvas ref={canvasRef} className="hidden" />
            {liveFrame && (
              <>
                <Dashboard summary={liveFrame.summary} compact={minimalUi} />
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={imageDataUrl(liveFrame.annotated_png_base64)}
                  alt="Live annotated"
                  className="max-h-[75vh] w-full rounded-md border object-contain"
                />
              </>
            )}
            {!liveFrame && liveOn && <p className="text-sm text-muted-foreground">Waiting for first frame…</p>}
          </div>
        )}
      </main>
    </div>
  );
}
