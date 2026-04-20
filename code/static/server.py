#!/usr/bin/env python3
import json, os, time, mimetypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HOST = "127.0.0.1"
PORT = 8000

# 데모용 클래스/확률
CLASS_NAMES = ['none','walk_away','walk_toward','sit','stand','squat','fall']

def fake_status_payload():
    # 2초 주기로 현재 액션 변경
    idx = int(time.time() // 2) % len(CLASS_NAMES)
    current = CLASS_NAMES[idx]

    # 확률 벡터(현재 액션에 가중치)
    probs = {name: 0.02 for name in CLASS_NAMES}
    probs[current] = 0.92
    s = sum(probs.values())
    probs = {k: round(v/s, 3) for k, v in probs.items()}

    # 카운트 대충 증가
    base = int(time.time() // 5)
    squat = base % 30
    stand = (base * 2) % 50

    # 진행 중 액션의 경과 시간
    now = time.time()
    t0 = int(now // 10) * 10
    elapsed = now - t0
    durations = {
        "walk_sec":        elapsed if "walk" in current else 0.0,
        "walk_toward_sec": elapsed if current == "walk_toward" else 0.0,
        "walk_away_sec":   elapsed if current == "walk_away"   else 0.0,
        "sit_sec":         elapsed if current == "sit"          else 0.0,
        "stand_sec":       elapsed if current == "stand"        else 0.0,
    }

    return {
        "squat_count": squat,
        "stand_count": stand,
        "current_action": current,
        "class_probs": probs,
        "durations": durations
    }

class Handler(SimpleHTTPRequestHandler):
    # 모든 응답에 공통 헤더 추가(CORS/캐시 무효화)
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        return super().end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)

        # /status 가짜 API
        if parsed.path == "/status":
            payload = fake_status_payload()
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
            return

        # 루트로 오면 main.html 서빙
        if parsed.path == "/":
            self.path = "/main.html"

        # 정적 파일 서빙(원래 핸들러 사용)
        # (mimetypes 초기화: mp3, css 등 올바른 MIME)
        if not mimetypes.inited:
            mimetypes.init()
        return super().do_GET()

def run():
    web_root = os.path.abspath(os.getcwd())
    with ThreadingHTTPServer((HOST, PORT), Handler) as httpd:
        print(f"Serving {web_root} at http://{HOST}:{PORT}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    run()
