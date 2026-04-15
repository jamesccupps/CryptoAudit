#!/usr/bin/env python3
"""
CryptoAudit ML — AI-Powered Encryption Analyzer
=================================================
Standalone GUI application. All data stored in this folder.

Folder structure:
    CryptoAudit/
    ├── crypto_audit.py        ← This file
    ├── CryptoAudit.bat        ← Windows launcher
    ├── models/                ← Trained ML models
    ├── feedback/samples/      ← Auto-labeled training data
    ├── logs/                  ← Debug + session logs
    ├── exports/               ← JSON/HTML session reports
    └── debug/                 ← Debug snapshots for troubleshooting
"""

import collections, hashlib, json, logging, math, os, re, struct, sys
import base64, csv, io, itertools, shutil, threading, time, traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import warnings
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, List

import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
#  App Directory — everything lives here
# ═══════════════════════════════════════════════════════════════
APP_DIR = Path(__file__).parent.resolve()
MODELS_DIR = APP_DIR / "models"
FEEDBACK_DIR = APP_DIR / "feedback"
SAMPLES_DIR = FEEDBACK_DIR / "samples"
LOGS_DIR = APP_DIR / "logs"
EXPORTS_DIR = APP_DIR / "exports"
DEBUG_DIR = APP_DIR / "debug"

for d in [MODELS_DIR, SAMPLES_DIR/"strong", SAMPLES_DIR/"moderate",
          SAMPLES_DIR/"weak", SAMPLES_DIR/"critical", LOGS_DIR, EXPORTS_DIR, DEBUG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  Logging — rotating file + console
# ═══════════════════════════════════════════════════════════════
log = logging.getLogger("CryptoAudit")
log.setLevel(logging.DEBUG)
log.propagate = False  # Don't pass to root logger (avoids cp1252 errors on Windows)
_fh = RotatingFileHandler(LOGS_DIR / "crypto_audit.log", maxBytes=5*1024*1024, backupCount=5, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_fh.setLevel(logging.DEBUG)
log.addHandler(_fh)
if sys.platform != 'win32':
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.WARNING)
    _ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(_ch)
log.info(f"CryptoAudit starting — app_dir={APP_DIR}")

# ═══════════════════════════════════════════════════════════════
#  Scapy (optional)
# ═══════════════════════════════════════════════════════════════
try:
    import scapy.utils6
    _orig = scapy.utils6.construct_source_candidate_set
    def _safe(p, pl, d):
        try: return _orig(p, pl, d)
        except: return []
    scapy.utils6.construct_source_candidate_set = _safe
    from scapy.utils import rdpcap, PcapReader
    from scapy.layers.inet import TCP, UDP, IP
    from scapy.packet import Raw
    HAS_SCAPY = True
    log.info("Scapy loaded")
except Exception as e:
    HAS_SCAPY = False
    log.warning(f"Scapy not available: {e}")

# ═══════════════════════════════════════════════════════════════
#  Comprehensive Port Map
# ═══════════════════════════════════════════════════════════════
PORT_MAP = {
    # TLS / HTTPS
    443: "HTTPS", 8443: "HTTPS-alt", 4443: "HTTPS-alt2", 8080: "HTTP-proxy",
    80: "HTTP", 8000: "HTTP-alt", 8888: "HTTP-alt2",
    # Email
    993: "IMAPS", 995: "POP3S", 465: "SMTPS", 587: "SMTP-TLS",
    # POS / Payment terminals
    5696: "POS-generic", 2050: "POS-NCR", 20000: "POS-Verifone",
    4430: "Payment-GW", 9100: "POS-print", 20002: "POS-Ingenico",
    443: "HTTPS", 5555: "POS-Pax",
    # Database (often carry encrypted credentials)
    3306: "MySQL", 5432: "PostgreSQL", 1433: "MSSQL", 1521: "Oracle",
    27017: "MongoDB", 6379: "Redis",
    # Remote access
    22: "SSH", 3389: "RDP", 5900: "VNC",
    # Directory / Auth
    636: "LDAPS", 389: "LDAP", 88: "Kerberos",
    # IoT / Building automation
    8883: "MQTT-TLS", 5671: "AMQP-TLS", 47808: "BACnet",
    # VPN
    500: "IKE", 4500: "IPSec-NAT", 1194: "OpenVPN", 51820: "WireGuard",
    # API / Webhooks
    9443: "API-TLS", 6443: "K8s-API", 2376: "Docker-TLS",
}

# Default BPF filter covering all interesting ports
DEFAULT_BPF = (
    "tcp port 443 or tcp port 8443 or tcp port 80 or tcp port 8080 "
    "or tcp port 993 or tcp port 995 or tcp port 465 or tcp port 636 "
    "or tcp port 5696 or tcp port 2050 or tcp port 20000 or tcp port 4430 "
    "or tcp port 3306 or tcp port 5432 or tcp port 1433 or tcp port 22 "
    "or tcp port 3389 or tcp port 8883 or tcp port 9443"
)

# ═══════════════════════════════════════════════════════════════
#  Feature Extraction (37 features)
# ═══════════════════════════════════════════════════════════════
class FeatureExtractor:
    NAMES = [
        "shannon_entropy","normalized_entropy","chi_square","unique_byte_ratio",
        "byte_range","byte_std","freq_skewness","freq_kurtosis","max_byte_freq",
        "min_byte_freq","median_byte_freq","freq_iqr","serial_corr_1","serial_corr_2",
        "serial_corr_4","runs_ratio","longest_run_ratio","block8_dup","block16_dup",
        "block32_dup","block8_max","block16_max","hamming_avg","ascii_ratio",
        "null_ratio","high_byte_ratio","low_nibble_ent","high_nibble_ent",
        "nibble_balance","bigram_entropy","trigram_unique","bigram_repeat_max",
        "bigram_chi","xor_ioc_best","xor_ioc_ratio","fft_peak","fft_period",
    ]

    @classmethod
    def extract(cls, data: bytes) -> np.ndarray:
        if len(data) < 16: return np.zeros(37)
        d=data[:8192]; n=len(d)
        arr=np.frombuffer(d,dtype=np.uint8).astype(np.float64); f=[]
        freq=np.bincount(arr.astype(int),minlength=256).astype(np.float64)
        probs=freq/n; pnz=probs[probs>0]
        shan=-np.sum(pnz*np.log2(pnz)); f.append(shan); f.append(shan/8.0)
        exp=n/256.0; chi=np.sum((freq-exp)**2/exp); f.append(chi)
        f.append(np.count_nonzero(freq)/256.0)
        f.append((np.max(arr)-np.min(arr))/255.0); f.append(np.std(arr)/128.0)
        fnz=freq[freq>0]
        f.append(float(skew(freq))); f.append(float(kurtosis(freq)))
        f.append(np.max(freq)/n)
        f.append(np.min(fnz)/n if len(fnz)>0 else 0); f.append(np.median(freq)/n)
        q75,q25=np.percentile(freq,[75,25]); f.append((q75-q25)/n)
        for lag in [1,2,4]:
            if n>lag+1:
                c=np.corrcoef(arr[:-lag],arr[lag:])[0,1]
                f.append(c if not np.isnan(c) else 0.0)
            else: f.append(0.0)
        bits=np.unpackbits(np.frombuffer(d[:512],dtype=np.uint8))
        runs=1+np.sum(bits[1:]!=bits[:-1]); f.append(runs/len(bits))
        mr=cr=1
        for i in range(1,len(bits)):
            if bits[i]==bits[i-1]: cr+=1; mr=max(mr,cr)
            else: cr=1
        f.append(mr/len(bits))
        for bs in [8,16,32]:
            bl=[d[i:i+bs] for i in range(0,n-bs+1,bs)]
            if bl: bc=collections.Counter(bl); f.append(sum(1 for c in bc.values() if c>1)/len(bc))
            else: f.append(0.0)
        for bs in [8,16]:
            bl=[d[i:i+bs] for i in range(0,n-bs+1,bs)]
            if bl: bc=collections.Counter(bl); f.append(bc.most_common(1)[0][1]/len(bl))
            else: f.append(0.0)
        b16=[arr[i:i+16] for i in range(0,n-16,16)]
        if len(b16)>1:
            hd=[np.sum(b16[i]!=b16[i+1])/16.0 for i in range(len(b16)-1)
                if len(b16[i])==16 and len(b16[i+1])==16]
            f.append(np.mean(hd) if hd else 0.5)
        else: f.append(0.5)
        f.append(sum(1 for b in d if 32<=b<=126)/n)
        f.append(d.count(0)/n); f.append(sum(1 for b in d if b>=128)/n)
        ln=arr.astype(int)&0x0F; hn=(arr.astype(int)>>4)&0x0F
        for nib in [ln,hn]:
            nf=np.bincount(nib,minlength=16).astype(np.float64)
            nv=nf/n; nnz=nv[nv>0]; f.append(-np.sum(nnz*np.log2(nnz)))
        f.append(abs(np.mean(ln)-np.mean(hn))/15.0)
        bg=[d[i:i+2] for i in range(min(n-1,4096))]
        bgf=collections.Counter(bg); bgt=len(bg)
        bp=np.array([c/bgt for c in bgf.values()]); bpnz=bp[bp>0]
        f.append(-np.sum(bpnz*np.log2(bpnz)))
        tg=[d[i:i+3] for i in range(min(n-2,4096))]
        f.append(len(collections.Counter(tg))/max(len(tg),1))
        f.append(bgf.most_common(1)[0][1]/bgt if bgf else 0)
        bge=bgt/65536.0; f.append(sum((c-bge)**2/bge for c in bgf.values())/bgt)
        xr={}
        for kl in range(1,min(33,n//4)):
            st=[[] for _ in range(kl)]
            for i,b in enumerate(d[:2048]): st[i%kl].append(b)
            iv=[]
            for s in st:
                if len(s)<2: continue
                sf=collections.Counter(s); sn=len(s)
                iv.append(sum(c*(c-1) for c in sf.values())/(sn*(sn-1)))
            if iv: xr[kl]=np.mean(iv)
        ri=1.0/256
        if xr: bi=max(xr.values()); f.append(bi); f.append(bi/ri)
        else: f.append(ri); f.append(1.0)
        if n>=64:
            sig=arr[:min(n,2048)]-np.mean(arr[:min(n,2048)])
            fm=np.abs(np.fft.rfft(sig)); fm[0]=0
            if len(fm)>1:
                pi=np.argmax(fm[1:])+1; ps=fm[pi]/(np.mean(fm[1:])+1e-10)
                f.append(min(ps/10.0,1.0)); f.append(pi/len(fm))
            else: f.extend([0.0,0.0])
        else: f.extend([0.0,0.0])
        return np.array(f,dtype=np.float64)

    @classmethod
    def extract_batch(cls, payloads): return np.array([cls.extract(p) for p in payloads])

# ═══════════════════════════════════════════════════════════════
#  Training Data Generator
# ═══════════════════════════════════════════════════════════════
class TrainingDataGenerator:
    LABELS={0:"STRONG",1:"MODERATE",2:"WEAK",3:"CRITICAL"}
    @classmethod
    def generate(cls, n=500):
        P=[]; L=[]
        for _ in range(n):
            sz=np.random.randint(128,4096)
            # STRONG: mix of pure random AND protocol-structured encryption
            strong_type=np.random.choice(["random","tls","ssh","structured"])
            if strong_type=="random":
                P.append(os.urandom(sz))
            elif strong_type=="tls":
                # Simulate TLS records: 5-byte header + encrypted payload
                buf=bytearray()
                while len(buf)<sz-10:
                    ct=np.random.choice([0x17,0x16,0x17,0x17])  # Mostly AppData
                    ver=np.random.choice([b'\x03\x03',b'\x03\x01',b'\x03\x04'])
                    remaining=sz-len(buf)-5
                    if remaining<16: break
                    rec_len=np.random.randint(16,min(16384,remaining)+1)
                    buf.append(ct); buf.extend(ver)
                    buf.extend(struct.pack("!H",rec_len))
                    buf.extend(os.urandom(rec_len))
                if len(buf)<sz: buf.extend(os.urandom(sz-len(buf)))
                P.append(bytes(buf[:sz]))
            elif strong_type=="ssh":
                # SSH-like: 4-byte length prefix + encrypted blocks
                buf=bytearray()
                while len(buf)<sz:
                    bl=np.random.randint(16,256)
                    buf.extend(struct.pack("!I",bl))
                    buf.extend(os.urandom(bl))
                P.append(bytes(buf[:sz]))
            else:
                # AES-CBC-like: high entropy but with IV prefix and block alignment
                iv=os.urandom(16)
                blocks=os.urandom(((sz-16)//16)*16)
                P.append((iv+blocks)[:sz])
            L.append(0)

            # MODERATE: random with minor imperfections
            d=bytearray(os.urandom(sz))
            for i in range(0,len(d),37):
                if i<len(d): d[i]=(d[i]&0xF0)|0x0A
            P.append(bytes(d)); L.append(1)

            # WEAK: actual weak encryption
            m=np.random.choice(["xor","ecb","caesar","sub","rc4"])
            pl=cls._p(sz)
            if m=="xor":
                k=os.urandom(np.random.randint(1,16))
                P.append(bytes(p^k[i%len(k)] for i,p in enumerate(pl)))
            elif m=="ecb":
                bl=os.urandom(16); r=np.random.uniform(0.3,0.7)
                P.append(b''.join(bl if np.random.random()<r else os.urandom(16) for _ in range(sz//16)))
            elif m=="caesar":
                s=np.random.randint(1,255); P.append(bytes((b+s)%256 for b in pl))
            elif m=="sub":
                t=list(range(256)); np.random.shuffle(t); P.append(bytes(t[b] for b in pl))
            else:
                k=os.urandom(np.random.randint(3,6)); s=list(range(256)); j=0
                for i in range(256): j=(j+s[i]+k[i%len(k)])%256; s[i],s[j]=s[j],s[i]
                o=bytearray(sz); ci=cj=0
                for x in range(sz):
                    ci=(ci+1)%256; cj=(cj+s[ci])%256; s[ci],s[cj]=s[cj],s[ci]
                    o[x]=pl[x]^s[(s[ci]+s[cj])%256]
                P.append(bytes(o))
            L.append(2)

            # CRITICAL: plaintext / trivially encoded
            m2=np.random.choice(["plain","b64","hex"])
            if m2=="plain": P.append(cls._p(sz))
            elif m2=="b64": P.append(base64.b64encode(cls._p(sz//2)))
            else: P.append(os.urandom(sz//2).hex().encode())
            L.append(3)
        return P, np.array(L)
    @staticmethod
    def _p(sz):
        t=[b'{"card":"4532015112830366","exp":"12/27","cvv":"123","amount":499.99}',
           b"POST /payment HTTP/1.1\r\ncard=4111111111111111&cvv=456&amt=299",
           b"TRANSACTION PAN=5425233430109903 EXP=0628 CVV=789 AMT=150.00"]
        b=t[np.random.randint(len(t))]; return(b*((sz//len(b))+2))[:sz]

# ═══════════════════════════════════════════════════════════════
#  ML Models
# ═══════════════════════════════════════════════════════════════
class CryptoMLModels:
    def __init__(self):
        self.scaler=StandardScaler(); self.rf=None; self.gb=None
        self.ae=None; self.iso=None; self.is_trained=False
        self.history=[]; self.feat_imp=None
        self._ae_thresh=0.0; self._ae_err=None

    def train(self, payloads, labels, cb=None):
        def _cb(m):
            log.info(m)
            if cb: cb(m)
        _cb(f"Extracting features from {len(payloads)} samples...")
        X=FeatureExtractor.extract_batch(payloads)
        X=np.nan_to_num(X,nan=0.0,posinf=1e6,neginf=-1e6)
        Xs=self.scaler.fit_transform(X)
        _cb("Training Random Forest (500 trees)...")
        self.rf=RandomForestClassifier(n_estimators=500,max_depth=20,min_samples_leaf=3,
            class_weight="balanced",n_jobs=-1,random_state=42)
        self.rf.fit(Xs,labels)
        rf_acc=cross_val_score(self.rf,Xs,labels,cv=5,scoring="accuracy").mean()
        _cb(f"Random Forest: {rf_acc:.1%}")
        _cb("Training Gradient Boosting...")
        self.gb=GradientBoostingClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,
            min_samples_leaf=5,random_state=42)
        self.gb.fit(Xs,labels)
        gb_acc=cross_val_score(self.gb,Xs,labels,cv=5,scoring="accuracy").mean()
        _cb(f"Gradient Boosting: {gb_acc:.1%}")
        _cb("Training Autoencoder...")
        Xst=Xs[labels==0]; nf=Xs.shape[1]
        self.ae=MLPRegressor(hidden_layer_sizes=(nf,16,8,16,nf),activation="relu",
            solver="adam",max_iter=500,learning_rate="adaptive",early_stopping=True,
            validation_fraction=0.15,random_state=42)
        self.ae.fit(Xst,Xst)
        rec=self.ae.predict(Xst)
        self._ae_err=np.mean((Xst-rec)**2,axis=1)
        self._ae_thresh=np.percentile(self._ae_err,95)
        _cb("Training Isolation Forest...")
        self.iso=IsolationForest(n_estimators=200,contamination=0.25,random_state=42,n_jobs=-1)
        self.iso.fit(Xst)
        self.feat_imp=dict(zip(FeatureExtractor.NAMES,self.rf.feature_importances_))
        self.is_trained=True
        self.history.append({"time":datetime.now().isoformat(),"n":len(payloads),
            "rf":float(rf_acc),"gb":float(gb_acc)})
        _cb("Training complete!")
        return {"rf":rf_acc,"gb":gb_acc}

    def predict(self, payload):
        if not self.is_trained:
            log.info("Auto-training on first use...")
            p,l=TrainingDataGenerator.generate(300)
            self.train(p,l); self.save()
        feat=FeatureExtractor.extract(payload)
        feat=np.nan_to_num(feat,nan=0.0,posinf=1e6,neginf=-1e6)
        X=self.scaler.transform(feat.reshape(1,-1))
        rfp=self.rf.predict_proba(X)[0]; gbp=self.gb.predict_proba(X)[0]
        rec=self.ae.predict(X); ae_err=float(np.mean((X-rec)**2))
        ae_anom=ae_err>self._ae_thresh; ae_sc=min(ae_err/(self._ae_thresh+1e-10),5.0)
        iso_sc=-float(self.iso.score_samples(X)[0])
        ep=0.6*rfp+0.4*gbp
        if ae_sc>2.0 and np.argmax(ep)==0: ep[0]*=0.5; ep/=ep.sum()
        pred=int(np.argmax(ep))
        return {"prediction":pred,"label":TrainingDataGenerator.LABELS.get(pred,"?"),
            "confidence":float(ep[pred]),
            "probs":{TrainingDataGenerator.LABELS[i]:float(p) for i,p in enumerate(ep) if i<4},
            "ae_score":round(ae_sc,4),"ae_anomaly":ae_anom,"iso_score":round(iso_sc,4),
            "triggers":self._trig(feat),
            "features":{n:round(float(v),6) for n,v in zip(FeatureExtractor.NAMES,feat)}}

    def _trig(self, f):
        t=[]; idx={n:i for i,n in enumerate(FeatureExtractor.NAMES)}
        e=f[idx["shannon_entropy"]]
        if e<5: t.append(f"Very low entropy ({e:.2f}/8.0)")
        elif e<7: t.append(f"Low entropy ({e:.2f}/8.0)")
        if f[idx["chi_square"]]>310: t.append(f"Chi-square failed ({f[idx['chi_square']]:.0f})")
        c=f[idx["serial_corr_1"]]
        if abs(c)>0.05: t.append(f"Serial correlation ({c:.4f})")
        if f[idx["block16_dup"]]>0.01: t.append(f"ECB repetition ({f[idx['block16_dup']]:.1%})")
        x=f[idx["xor_ioc_ratio"]]
        if x>3: t.append(f"XOR pattern (IoC {x:.1f}x)")
        a=f[idx["ascii_ratio"]]
        if a>0.7: t.append(f"High ASCII ({a:.1%}) — likely plaintext")
        if f[idx["null_ratio"]]>0.1: t.append(f"Null bytes ({f[idx['null_ratio']]:.1%})")
        if f[idx["fft_peak"]]>0.3: t.append(f"Periodicity (FFT {f[idx['fft_peak']]:.2f})")
        return t

    def save(self):
        joblib.dump({"scaler":self.scaler,"rf":self.rf,"gb":self.gb,"ae":self.ae,
            "iso":self.iso,"ae_thresh":self._ae_thresh,"ae_err":self._ae_err,
            "feat_imp":self.feat_imp,"history":self.history,"trained":self.is_trained},
            MODELS_DIR/"models.joblib")
        log.info("Models saved")

    def load(self):
        p=MODELS_DIR/"models.joblib"
        if not p.exists(): return False
        try:
            s=joblib.load(p)
            self.scaler=s["scaler"]; self.rf=s["rf"]; self.gb=s["gb"]
            self.ae=s["ae"]; self.iso=s["iso"]; self._ae_thresh=s["ae_thresh"]
            self._ae_err=s["ae_err"]; self.feat_imp=s["feat_imp"]
            self.history=s.get("history",[]); self.is_trained=s["trained"]
            log.info("Models loaded"); return True
        except Exception as e:
            log.error(f"Model load failed: {e}"); return False

# ═══════════════════════════════════════════════════════════════
#  PAN Detector + Decryption Engine + AI Validator + Feedback
#  (same as before but with proper logging)
# ═══════════════════════════════════════════════════════════════
class PANDetector:
    PATS={"visa":re.compile(rb'4[0-9]{12}(?:[0-9]{3})?'),
          "mc":re.compile(rb'5[1-5][0-9]{14}'),
          "amex":re.compile(rb'3[47][0-9]{13}')}
    @staticmethod
    def luhn(n):
        d=[int(c) for c in n if c.isdigit()]
        if len(d)<13: return False
        cs=0
        for i,v in enumerate(reversed(d)):
            if i%2==1: v*=2
            if v>9: v-=9
            cs+=v
        return cs%10==0
    @classmethod
    def scan(cls, data):
        hits=[]
        for ct,pat in cls.PATS.items():
            for m in pat.finditer(data):
                ns=m.group().decode('ascii',errors='ignore')
                if cls.luhn(ns):
                    hits.append({"type":"PAN","card":ct,"off":m.start(),
                        "masked":ns[:6]+"******"+ns[-4:]})
        return hits

class ProtocolDetector:
    """Identifies encrypted protocol traffic that SHOULD be opaque."""

    TLS_PORTS={443,8443,993,995,465,636,853,9443,6443,2376}
    SSH_PORTS={22}

    @staticmethod
    def detect(data: bytes, sport: int = 0, dport: int = 0) -> dict:
        """Returns protocol info. Uses ports as fallback when headers aren't at offset 0."""
        if len(data) < 5:
            return {"protocol": None, "encrypted_properly": False}
        b0 = data[0]

        # TLS record header at offset 0
        if b0 in (0x14, 0x15, 0x16, 0x17) and len(data) >= 5:
            ver = struct.unpack("!H", data[1:3])[0]
            if ver in (0x0300, 0x0301, 0x0302, 0x0303, 0x0304):
                tls_versions = {0x0300:"SSL3.0",0x0301:"TLS1.0",0x0302:"TLS1.1",0x0303:"TLS1.2",0x0304:"TLS1.3"}
                return {"protocol": "TLS", "version": tls_versions.get(ver, f"0x{ver:04x}"),
                        "record_type": {0x14:"ChangeCipherSpec",0x15:"Alert",0x16:"Handshake",0x17:"AppData"}.get(b0,"Unknown"),
                        "encrypted_properly": b0 == 0x17 or ver >= 0x0303,
                        "weak_version": ver < 0x0303}

        # TLS record boundaries inside data (mid-stream capture)
        tls_markers = 0
        for i in range(0, min(len(data)-5, 4096)):
            if data[i] in (0x16, 0x17) and data[i+1:i+3] in (b'\x03\x01', b'\x03\x03', b'\x03\x04'):
                rec_len = struct.unpack("!H", data[i+3:i+5])[0]
                if 1 <= rec_len <= 16384:  # Valid TLS record length
                    tls_markers += 1
        if tls_markers >= 2:
            return {"protocol": "TLS", "version": "mixed",
                    "encrypted_properly": True, "markers_found": tls_markers}

        # Port-based TLS fallback — high entropy on a TLS port = almost certainly TLS
        if (sport in ProtocolDetector.TLS_PORTS or dport in ProtocolDetector.TLS_PORTS):
            sample = data[:1024]
            ent_approx = len(set(sample)) / 256
            if ent_approx > 0.6 and len(data) >= 64:
                return {"protocol": "TLS", "version": "inferred",
                        "encrypted_properly": True, "port_inferred": True}

        # SSH
        if data[:4] == b'SSH-':
            return {"protocol": "SSH", "version": data[:20].decode('ascii',errors='ignore').strip(),
                    "encrypted_properly": False}  # Banner is plaintext
        if sport in ProtocolDetector.SSH_PORTS or dport in ProtocolDetector.SSH_PORTS:
            if len(data) >= 64:
                sample = data[:512]
                ent_approx = len(set(sample)) / 256
                if ent_approx > 0.5:
                    return {"protocol": "SSH", "version": "encrypted",
                            "encrypted_properly": True, "port_inferred": True}

        # STUN/TURN (ports 3478, 3479, 19302)
        if sport in (3478,3479,19302) or dport in (3478,3479,19302):
            if len(data) >= 20:
                msg_type = struct.unpack("!H", data[0:2])[0]
                msg_len = struct.unpack("!H", data[2:4])[0]
                magic = data[4:8]
                if magic == b'\x21\x12\xa4\x42':  # STUN magic cookie
                    stun_types = {0x0001:"Binding Request",0x0101:"Binding Response",
                                  0x0003:"Allocate",0x0103:"Allocate Response",
                                  0x0016:"Data",0x0116:"Data Response"}
                    return {"protocol": "STUN", "version": stun_types.get(msg_type,f"0x{msg_type:04x}"),
                            "encrypted_properly": False, "msg_len": msg_len}

        # WireGuard
        if b0 in (1,2,3,4) and len(data) >= 32:
            if (b0 == 1 and len(data) >= 148) or (b0 == 2 and len(data) >= 92):
                return {"protocol": "WireGuard", "encrypted_properly": True}

        return {"protocol": None, "encrypted_properly": False}

    @staticmethod
    def is_dns(data: bytes, sport: int = 0, dport: int = 0) -> bool:
        """Check if data looks like a DNS query/response."""
        if sport == 53 or dport == 53 or sport == 853 or dport == 853:
            return True
        # DNS has: 2-byte ID, 2-byte flags, 4x 2-byte counts, then questions
        if len(data) >= 12:
            flags = struct.unpack("!H", data[2:4])[0]
            qr = (flags >> 15) & 1  # 0=query, 1=response
            opcode = (flags >> 11) & 0xF
            qdcount = struct.unpack("!H", data[4:6])[0]
            if opcode <= 2 and qdcount > 0 and qdcount < 50:
                return True
        return False

    @staticmethod
    def is_binary_noise(data: bytes) -> bool:
        """Check if data is high-entropy binary that shouldn't be text-analyzed."""
        if len(data) < 32: return False
        sample = data[:1024]
        printable = sum(1 for b in sample if 32 <= b <= 126) / len(sample)
        null_ratio = sample.count(0) / len(sample)
        high_bytes = sum(1 for b in sample if b >= 128) / len(sample)
        return printable < 0.4 and high_bytes > 0.3 and null_ratio < 0.1


class DNSDecoder:
    """Parse DNS query/response packets to extract domain names and record data."""

    @staticmethod
    def decode(data: bytes) -> dict:
        """Parse a DNS packet and extract human-readable content."""
        if len(data) < 12: return None
        try:
            txid = struct.unpack("!H", data[0:2])[0]
            flags = struct.unpack("!H", data[2:4])[0]
            qr = (flags >> 15) & 1
            rcode = flags & 0xF
            qdcount = struct.unpack("!H", data[4:6])[0]
            ancount = struct.unpack("!H", data[6:8])[0]

            domains = []
            answers = []
            pos = 12

            # Parse question section
            for _ in range(min(qdcount, 10)):
                name, pos = DNSDecoder._read_name(data, pos)
                if name and pos + 4 <= len(data):
                    qtype = struct.unpack("!H", data[pos:pos+2])[0]
                    pos += 4  # skip qtype + qclass
                    type_name = {1:"A",2:"NS",5:"CNAME",6:"SOA",12:"PTR",15:"MX",
                                 16:"TXT",28:"AAAA",33:"SRV",65:"HTTPS"}.get(qtype, f"TYPE{qtype}")
                    domains.append({"name": name, "type": type_name, "section": "question"})

            # Parse answer section
            for _ in range(min(ancount, 20)):
                if pos >= len(data) - 4: break
                name, pos = DNSDecoder._read_name(data, pos)
                if pos + 10 > len(data): break
                atype, aclass, ttl, rdlen = struct.unpack("!HHIH", data[pos:pos+10])
                pos += 10
                rdata_raw = data[pos:pos+rdlen] if pos+rdlen <= len(data) else b''
                pos += rdlen

                type_name = {1:"A",2:"NS",5:"CNAME",6:"SOA",12:"PTR",15:"MX",
                             16:"TXT",28:"AAAA",33:"SRV",65:"HTTPS"}.get(atype, f"TYPE{atype}")
                rdata = ""
                if atype == 1 and len(rdata_raw) == 4:  # A record
                    rdata = f"{rdata_raw[0]}.{rdata_raw[1]}.{rdata_raw[2]}.{rdata_raw[3]}"
                elif atype == 28 and len(rdata_raw) == 16:  # AAAA
                    rdata = ":".join(f"{rdata_raw[i]:02x}{rdata_raw[i+1]:02x}" for i in range(0,16,2))
                elif atype in (2, 5, 12):  # NS, CNAME, PTR
                    rdata, _ = DNSDecoder._read_name(data, pos-rdlen)
                elif atype == 16:  # TXT
                    rdata = rdata_raw[1:].decode('ascii', errors='ignore') if rdata_raw else ""

                answers.append({"name": name or "?", "type": type_name, "ttl": ttl,
                                "data": rdata, "section": "answer"})

            if not domains and not answers: return None

            return {
                "method": "DNS Decode", "success": True,
                "txid": f"0x{txid:04x}",
                "qr": "response" if qr else "query",
                "rcode": {0:"OK",1:"FORMAT_ERR",2:"SERVER_FAIL",3:"NXDOMAIN",5:"REFUSED"}.get(rcode, f"RCODE{rcode}"),
                "domains": domains,
                "answers": answers,
                "query_names": [d["name"] for d in domains],
                "resolved_ips": [a["data"] for a in answers if a["type"] in ("A","AAAA") and a["data"]],
                "confidence": 0.95,
                "printable": 0.0,
            }
        except Exception as e:
            log.debug(f"DNS decode failed: {e}")
            return None

    @staticmethod
    def _read_name(data, pos):
        """Read a DNS name with pointer compression."""
        parts = []; jumps = 0; max_jumps = 10
        while pos < len(data) and jumps < max_jumps:
            length = data[pos]
            if length == 0:
                pos += 1; break
            elif (length & 0xC0) == 0xC0:  # Pointer
                if pos + 1 >= len(data): break
                ptr = struct.unpack("!H", data[pos:pos+2])[0] & 0x3FFF
                pos += 2
                # Follow pointer but don't update pos for return
                sub_name, _ = DNSDecoder._read_name(data, ptr)
                if sub_name: parts.append(sub_name)
                return '.'.join(parts) if parts else None, pos
            else:
                pos += 1
                if pos + length > len(data): break
                if length > 63: break  # DNS labels max 63 chars
                label = data[pos:pos+length]
                # Validate: DNS labels must be printable ASCII (letters, digits, hyphens)
                if not all(32 <= b <= 126 for b in label):
                    return None, pos + length  # Non-printable = not a valid DNS name
                parts.append(label.decode('ascii', errors='ignore'))
                pos += length
            jumps += 1
        return '.'.join(parts) if parts else None, pos


class PlaintextProtocolAnalyzer:
    """Detects cleartext protocols and extracts credentials/sensitive data.
    These protocols send data with NO encryption — the 'decryption' is just reading."""

    PROTOCOL_SIGS = {
        'HTTP': {'ports': {80,8080,8000,8888,8008}, 'markers': [b'HTTP/',b'GET ',b'POST ',b'PUT ',b'DELETE ',b'HEAD ',b'PATCH ']},
        'FTP': {'ports': {21}, 'markers': [b'220 ',b'USER ',b'PASS ',b'230 ',b'530 ',b'RETR ',b'STOR ',b'LIST ']},
        'Telnet': {'ports': {23}, 'markers': [b'login:',b'Password:',b'Username:']},  # No IAC bytes — too many FP on TLS
        'SMTP': {'ports': {25,587}, 'markers': [b'EHLO ',b'HELO ',b'MAIL FROM:',b'RCPT TO:',b'AUTH ']},
        'SNMP': {'ports': {161,162}, 'markers': [b'\x30',b'public',b'private']},
        'MQTT': {'ports': {1883}, 'markers': [b'\x10',b'MQTT']},
        'Modbus': {'ports': {502}, 'markers': []},
        'SIP': {'ports': {5060}, 'markers': [b'SIP/2.0',b'INVITE ',b'REGISTER ',b'BYE ']},
        'RTSP': {'ports': {554}, 'markers': [b'RTSP/',b'DESCRIBE ',b'SETUP ',b'PLAY ']},
        'LDAP': {'ports': {389}, 'markers': [b'\x30']},
        'WeatherFlow': {'ports': {50222}, 'markers': [b'serial_number',b'"type"',b'obs_st',b'obs_air',b'rapid_wind',b'evt_precip']},
        'SSDP': {'ports': {1900}, 'markers': [b'NOTIFY ',b'M-SEARCH ',b'HTTP/1.1 200',b'ssdp:']},
    }

    # Ports where Telnet/HTTP marker matching should be suppressed (TLS traffic)
    TLS_PORTS = {443,8443,993,995,465,636,853,9443,6443,2376}

    @classmethod
    def analyze(cls, data, sport=0, dport=0):
        """Detect cleartext protocols and extract credentials/data."""
        if len(data) < 8: return None

        # Two-pass: port-matched protocols first (high confidence), then marker-only
        for require_port in (True, False):
            for proto_name, sig in cls.PROTOCOL_SIGS.items():
                port_match = sport in sig['ports'] or dport in sig['ports']
                marker_match = any(m in data[:512] for m in sig['markers']) if sig['markers'] else False

                if require_port and not port_match: continue
                if not require_port and port_match: continue  # Already checked in pass 1
                if not port_match and not marker_match: continue

                # NEVER match Telnet or HTTP by marker alone on TLS ports
                is_tls_port = sport in cls.TLS_PORTS or dport in cls.TLS_PORTS
                if is_tls_port and not port_match and proto_name in ('Telnet', 'HTTP'):
                    continue

                # Generic single-byte markers MUST have port match to avoid false positives
                generic_marker_protos = {'SNMP', 'LDAP', 'MQTT', 'Modbus'}
                if proto_name in generic_marker_protos and not port_match: continue

                # FTP '220 ' is too generic — require port OR multiple FTP markers
                if proto_name == 'FTP' and not port_match:
                    ftp_markers = sum(1 for m in sig['markers'] if m in data[:512])
                    if ftp_markers < 2: continue

                # Binary protocols need structural validation beyond just port
                if port_match and not marker_match and proto_name in ('SNMP', 'LDAP', 'Modbus'):
                    if proto_name == 'Modbus' and len(data) >= 12:
                        proto_id = struct.unpack("!H", data[2:4])[0]
                        if proto_id != 0: continue
                    elif proto_name == 'SNMP' and data[0] == 0x30:
                        pass  # ASN.1 sequence on SNMP port
                    else:
                        continue

                # Found a cleartext protocol — extract what we can
                result = {"method": "Plaintext Protocol", "success": True,
                          "protocol": proto_name, "port_match": port_match,
                          "printable": 0.0}

                if proto_name == 'HTTP':
                    result.update(cls._parse_http(data))
                    # Validate HTTP content is actually readable
                    preview = result.get("preview", "")
                    min_printable = 0.6 if port_match else 0.8  # Stricter for marker-only
                    if preview and sum(1 for c in preview[:100] if 32<=ord(c)<=126)/max(len(preview[:100]),1) < min_printable:
                        continue  # Garbled — likely encrypted payload over HTTP
                elif proto_name == 'FTP': result.update(cls._parse_ftp(data))
                elif proto_name == 'Telnet': result.update(cls._parse_telnet(data))
                elif proto_name == 'SMTP': result.update(cls._parse_smtp(data))
                elif proto_name == 'SNMP': result.update(cls._parse_snmp(data))
                elif proto_name == 'MQTT': result.update(cls._parse_mqtt(data))
                elif proto_name == 'Modbus': result.update(cls._parse_modbus(data))
                elif proto_name in ('SIP', 'RTSP'): result.update(cls._parse_sip(data, proto_name))
                elif proto_name == 'WeatherFlow': result.update(cls._parse_weatherflow(data))
                elif proto_name == 'SSDP': result.update(cls._parse_ssdp(data))

                result["confidence"] = 0.9 if port_match and marker_match else 0.7 if port_match else 0.6
                return result
        return None

    @classmethod
    def _parse_http(cls, data):
        """Extract HTTP headers, auth credentials, form data."""
        text = data[:8192].decode('ascii', errors='ignore')
        info = {"headers": {}, "credentials": [], "sensitive_fields": []}

        # Extract headers
        lines = text.split('\r\n')
        info["request_line"] = lines[0][:120] if lines else ""
        for line in lines[1:]:
            if ':' in line:
                k, v = line.split(':', 1)
                info["headers"][k.strip().lower()] = v.strip()[:100]
            if not line: break  # End of headers

        # HTTP Basic Auth — base64-encoded credentials
        auth = info["headers"].get("authorization", "")
        if auth.lower().startswith("basic "):
            try:
                decoded = base64.b64decode(auth[6:].strip()).decode('utf-8', errors='ignore')
                if ':' in decoded:
                    user, pwd = decoded.split(':', 1)
                    info["credentials"].append({"type": "HTTP_BASIC", "username": user, "password": pwd})
            except: pass
        # Digest auth
        if auth.lower().startswith("digest "):
            info["credentials"].append({"type": "HTTP_DIGEST", "header": auth[:100]})

        # POST body credentials
        body_start = text.find('\r\n\r\n')
        if body_start > 0:
            body = text[body_start+4:]
            for pattern in [r'(?:user(?:name)?|login|email)=([^&\s]{1,60})',
                           r'(?:pass(?:word)?|pwd|secret)=([^&\s]{1,60})']:
                for m in re.finditer(pattern, body, re.I):
                    info["sensitive_fields"].append(m.group(0))

        # Cookie session tokens
        cookie = info["headers"].get("cookie", "")
        if cookie:
            for tok in ['session', 'token', 'auth', 'jwt', 'sid']:
                if tok in cookie.lower():
                    info["sensitive_fields"].append(f"Cookie contains '{tok}'")
                    break

        info["preview"] = text[:200]
        info["printable"] = sum(1 for c in text[:500] if 32<=ord(c)<=126)/max(len(text[:500]),1)
        info["note"] = "HTTP cleartext — all data readable"
        return info

    @classmethod
    def _parse_ftp(cls, data):
        text = data[:4096].decode('ascii', errors='ignore')
        creds = []
        for m in re.finditer(r'USER\s+(\S+)', text): creds.append({"type":"FTP_USER","value":m.group(1)})
        for m in re.finditer(r'PASS\s+(\S+)', text): creds.append({"type":"FTP_PASS","value":m.group(1)})
        return {"credentials": creds, "preview": text[:200],
                "printable": 0.9, "note": "FTP cleartext — credentials visible"}

    @classmethod
    def _parse_telnet(cls, data):
        # Filter out telnet negotiation bytes (IAC sequences)
        clean = re.sub(b'\xff[\xfb-\xfe].', b'', data[:2048])
        text = clean.decode('ascii', errors='ignore').strip()
        return {"preview": text[:200], "printable": 0.8,
                "note": "Telnet cleartext — keystrokes/commands visible"}

    @classmethod
    def _parse_smtp(cls, data):
        text = data[:4096].decode('ascii', errors='ignore')
        creds = []
        # AUTH PLAIN sends base64(user\x00user\x00pass)
        for m in re.finditer(r'AUTH PLAIN\s+(\S+)', text):
            try:
                decoded = base64.b64decode(m.group(1)).decode('utf-8', errors='ignore')
                parts = decoded.split('\x00')
                if len(parts) >= 3: creds.append({"type":"SMTP_AUTH","user":parts[1],"pass":parts[2]})
            except: pass
        # AUTH LOGIN sends base64 user then base64 pass on next lines
        emails = re.findall(r'MAIL FROM:<([^>]+)>|RCPT TO:<([^>]+)>', text)
        return {"credentials": creds, "emails": [e[0] or e[1] for e in emails],
                "preview": text[:200], "printable": 0.9, "note": "SMTP cleartext"}

    @classmethod
    def _parse_snmp(cls, data):
        # SNMP v1/v2c: community string is in the clear
        # Basic ASN.1 parsing: SEQUENCE > version > community_string
        community = ""
        if len(data) > 10 and data[0] == 0x30:  # SEQUENCE
            try:
                pos = 2
                if data[1] & 0x80: pos += (data[1] & 0x7f)  # Long form length
                # Skip version INTEGER
                if data[pos] == 0x02:  # INTEGER
                    vlen = data[pos+1]; pos += 2 + vlen
                # Community string OCTET STRING
                if pos < len(data) and data[pos] == 0x04:
                    clen = data[pos+1]; pos += 2
                    community = data[pos:pos+clen].decode('ascii', errors='ignore')
            except: pass
        return {"community_string": community, "printable": 0.8,
                "preview": f"SNMP community='{community}'" if community else "SNMP",
                "note": f"SNMP v1/v2c — community string: '{community}'" if community else "SNMP traffic"}

    @classmethod
    def _parse_mqtt(cls, data):
        # MQTT CONNECT: byte0=0x10, then remaining length, then "MQTT" protocol name
        info = {"topics": [], "printable": 0.7}
        if data[0] == 0x10 and b'MQTT' in data[:20]:
            # Extract client ID from CONNECT
            try:
                mqtt_pos = data.index(b'MQTT')
                pos = mqtt_pos + 4 + 4  # Skip protocol level + connect flags + keepalive
                if pos + 2 < len(data):
                    cid_len = struct.unpack("!H", data[pos:pos+2])[0]
                    cid = data[pos+2:pos+2+cid_len].decode('ascii', errors='ignore')
                    info["client_id"] = cid
            except: pass
        # PUBLISH: byte0 high nibble = 0x3
        if (data[0] >> 4) == 3:
            try:
                rem_len = data[1]; pos = 2
                topic_len = struct.unpack("!H", data[pos:pos+2])[0]; pos += 2
                topic = data[pos:pos+topic_len].decode('ascii', errors='ignore')
                info["topics"].append(topic)
                payload = data[pos+topic_len:]
                info["preview"] = f"MQTT topic='{topic}' payload={payload[:80]}"
            except: pass
        info["note"] = "MQTT cleartext — IoT messages readable"
        return info

    @classmethod
    def _parse_modbus(cls, data):
        # Modbus TCP: 7-byte header + PDU
        if len(data) < 8: return {"note": "Modbus TCP"}
        txn_id, proto_id, length, unit_id = struct.unpack("!HHHB", data[:7])
        func_code = data[7] if len(data) > 7 else 0
        func_names = {1:"Read Coils",2:"Read Discrete",3:"Read Holding Registers",
                      4:"Read Input Registers",5:"Write Single Coil",6:"Write Single Register",
                      15:"Write Multiple Coils",16:"Write Multiple Registers"}
        return {"unit_id": unit_id, "function": func_names.get(func_code, f"0x{func_code:02x}"),
                "function_code": func_code, "printable": 0.5,
                "preview": f"Modbus unit={unit_id} func={func_names.get(func_code,func_code)}",
                "note": "Modbus TCP — industrial control, ZERO encryption"}

    @classmethod
    def _parse_sip(cls, data, proto="SIP"):
        text = data[:4096].decode('ascii', errors='ignore')
        creds = []
        # SIP Digest auth
        for m in re.finditer(r'Authorization:\s*Digest\s+(.+?)(?:\r?\n|$)', text):
            creds.append({"type":"SIP_DIGEST","header":m.group(1)[:100]})
        # Extract URIs
        uris = re.findall(r'sip:([^>\s;]+)', text)
        return {"credentials": creds, "uris": uris[:5],
                "preview": text[:200], "printable": 0.9,
                "note": f"{proto} cleartext — call signaling visible"}

    @classmethod
    def _parse_weatherflow(cls, data):
        """Parse WeatherFlow Tempest UDP broadcast (JSON on port 50222)."""
        text = data[:4096].decode('utf-8', errors='ignore')
        info = {"printable": 0.95, "note": "WeatherFlow Tempest broadcast — plaintext JSON"}
        try:
            # May contain multiple JSON objects concatenated
            obj = json.loads(text.split('\n')[0] if '\n' in text else text)
            info["device_serial"] = obj.get("serial_number", "")
            info["msg_type"] = obj.get("type", "")
            info["hub_sn"] = obj.get("hub_sn", "")
            if "obs" in obj:
                info["observations"] = len(obj["obs"])
            info["preview"] = f"WeatherFlow {info['msg_type']} from {info['device_serial']}"
            if info["hub_sn"]:
                info["preview"] += f" (hub: {info['hub_sn']})"
        except:
            info["preview"] = text[:200]
        return info

    @classmethod
    def _parse_ssdp(cls, data):
        """Parse SSDP/UPnP discovery and notification messages."""
        text = data[:4096].decode('ascii', errors='ignore')
        info = {"printable": 0.95}
        # Extract key headers
        location = ""
        server = ""
        usn = ""
        for line in text.split('\r\n'):
            ll = line.lower()
            if ll.startswith('location:'): location = line.split(':', 1)[1].strip()
            elif ll.startswith('server:'): server = line.split(':', 1)[1].strip()
            elif ll.startswith('usn:'): usn = line.split(':', 1)[1].strip()
        info["location"] = location
        info["server"] = server
        info["usn"] = usn
        info["preview"] = text[:200]
        info["note"] = f"SSDP/UPnP — {server or 'device discovery'}"
        if location:
            info["note"] += f" at {location}"
        return info


class TLSHandshakeParser:
    """Parse TLS ClientHello/ServerHello for intelligence WITHOUT decryption.
    Extracts: SNI hostname, cipher suites, certificate info, JA3 fingerprint."""

    @staticmethod
    def parse(data):
        """Extract TLS handshake info if present."""
        if len(data) < 10: return None

        # Find ClientHello (0x16 0x03 0x0X 0xLL 0xLL 0x01)
        for offset in range(min(len(data)-10, 4096)):
            if (data[offset] == 0x16 and data[offset+1] == 0x03 
                and data[offset+2] in (0x00,0x01,0x02,0x03,0x04)
                and offset+5 < len(data) and data[offset+5] == 0x01):
                return TLSHandshakeParser._parse_client_hello(data[offset:])

            # ServerHello (handshake type 0x02)
            if (data[offset] == 0x16 and data[offset+1] == 0x03
                and data[offset+2] in (0x00,0x01,0x02,0x03,0x04)
                and offset+5 < len(data) and data[offset+5] == 0x02):
                return TLSHandshakeParser._parse_server_hello(data[offset:])

            # Certificate (handshake type 0x0b)
            if (data[offset] == 0x16 and data[offset+1] == 0x03
                and data[offset+2] in (0x00,0x01,0x02,0x03,0x04)
                and offset+5 < len(data) and data[offset+5] == 0x0b):
                return TLSHandshakeParser._parse_certificate(data[offset:])
        return None

    @staticmethod
    def _parse_client_hello(data):
        """Extract SNI, cipher suites, JA3 from ClientHello."""
        result = {"method": "TLS Handshake", "success": True, "type": "ClientHello",
                  "confidence": 0.9, "printable": 0.0}
        try:
            # Record header: type(1) + version(2) + length(2)
            # Handshake header: type(1) + length(3)
            # ClientHello: version(2) + random(32) + session_id_len(1) + session_id
            pos = 5 + 4 + 2 + 32  # Skip to session_id_len
            if pos >= len(data): return result
            sid_len = data[pos]; pos += 1 + sid_len

            # Cipher suites
            if pos + 2 >= len(data): return result
            cs_len = struct.unpack("!H", data[pos:pos+2])[0]; pos += 2
            suites = []
            for i in range(0, min(cs_len, 200), 2):
                if pos + i + 2 > len(data): break
                suites.append(struct.unpack("!H", data[pos+i:pos+i+2])[0])
            pos += cs_len

            # Compression methods
            if pos >= len(data): return result
            comp_len = data[pos]; pos += 1 + comp_len

            # Extensions
            sni = ""
            if pos + 2 < len(data):
                ext_len = struct.unpack("!H", data[pos:pos+2])[0]; pos += 2
                ext_end = min(pos + ext_len, len(data))
                while pos + 4 < ext_end:
                    ext_type = struct.unpack("!H", data[pos:pos+2])[0]
                    ext_data_len = struct.unpack("!H", data[pos+2:pos+4])[0]
                    pos += 4
                    if ext_type == 0 and ext_data_len > 5:  # SNI
                        # SNI list length(2) + type(1) + name_len(2) + name
                        name_len = struct.unpack("!H", data[pos+3:pos+5])[0]
                        sni = data[pos+5:pos+5+name_len].decode('ascii', errors='ignore')
                    pos += ext_data_len

            # Weak cipher detection
            WEAK_CIPHERS = {0x000A,0x0013,0x0004,0x0005,0x002F,0x0035}  # RC4, DES, export
            weak = [s for s in suites if s in WEAK_CIPHERS]

            # JA3 fingerprint (simplified)
            ja3_str = f"771,{'-'.join(str(s) for s in suites[:30])}"
            ja3 = hashlib.md5(ja3_str.encode()).hexdigest()

            result["sni"] = sni
            result["cipher_suites"] = len(suites)
            result["weak_ciphers"] = len(weak)
            result["ja3"] = ja3
            result["note"] = f"SNI: {sni}" if sni else "ClientHello (no SNI)"
            if weak: result["note"] += f" — ⚠ {len(weak)} weak ciphers offered"
        except: pass
        return result

    @staticmethod
    def _parse_server_hello(data):
        """Extract chosen cipher suite from ServerHello."""
        result = {"method": "TLS Handshake", "success": True, "type": "ServerHello",
                  "confidence": 0.9, "printable": 0.0}
        try:
            pos = 5 + 4 + 2 + 32  # Skip to session_id_len
            if pos >= len(data): return result
            sid_len = data[pos]; pos += 1 + sid_len
            if pos + 2 > len(data): return result
            chosen = struct.unpack("!H", data[pos:pos+2])[0]
            CIPHER_NAMES = {0x1301:"TLS_AES_128_GCM_SHA256",0x1302:"TLS_AES_256_GCM_SHA384",
                0x1303:"TLS_CHACHA20_POLY1305",0xC02F:"ECDHE-RSA-AES128-GCM",
                0xC030:"ECDHE-RSA-AES256-GCM",0x002F:"RSA-AES128-CBC",0x0035:"RSA-AES256-CBC",
                0xC013:"ECDHE-RSA-AES128-SHA",0xC014:"ECDHE-RSA-AES256-SHA"}
            result["chosen_cipher"] = CIPHER_NAMES.get(chosen, f"0x{chosen:04x}")
            result["cipher_code"] = chosen
            result["note"] = f"Server chose {result['chosen_cipher']}"
            if chosen in (0x002F,0x0035): result["note"] += " — ⚠ No forward secrecy (RSA key exchange)"
        except: pass
        return result

    @staticmethod
    def _parse_certificate(data):
        """Extract certificate subject/issuer from Certificate message."""
        result = {"method": "TLS Handshake", "success": True, "type": "Certificate",
                  "confidence": 0.8, "printable": 0.0, "subjects": []}
        try:
            # Look for common certificate OID patterns in the raw data
            # CN (Common Name) OID: 2.5.4.3 = 55 04 03
            for m in re.finditer(b'\x55\x04\x03.(.{1,64}?)[\x30\x31]', data[:8192]):
                cn_data = m.group(1)
                # Skip the tag/length byte
                if len(cn_data) > 2:
                    cn = cn_data[1:].decode('ascii', errors='ignore').strip('\x00')
                    cn = ''.join(c for c in cn if 32 <= ord(c) <= 126)
                    if len(cn) > 3 and '.' in cn:
                        result["subjects"].append(cn)
            result["subjects"] = list(dict.fromkeys(result["subjects"]))[:5]  # Dedupe
            if result["subjects"]:
                result["note"] = f"Cert: {', '.join(result['subjects'][:3])}"
        except: pass
        return result


class DecryptionEngine:
    @staticmethod
    def _pr(d): return sum(1 for b in d if 32<=b<=126 or b in(9,10,13))/max(len(d),1)
    @staticmethod
    def _st(d):
        sc=0
        for p in [b'HTTP',b'GET ',b'POST',b'card',b'CARD',b'cvv',b'amount',b'<?xml',
                   b'{"',b'password',b'auth',b'token',b'=',b'&',b'SELECT',b'INSERT',
                   b'<!DOCTYPE',b'<html',b'BEGIN',b'END',b'function',b'var ',b'const ']:
            if p in d[:4096]: sc+=0.08
        return min(sc,1.0)
    @staticmethod
    def _prev(d, mx=200):
        return re.sub(r'(\d{6})\d{6,10}(\d{4})',r'\1******\2',
            ''.join(chr(b) if 32<=b<=126 else '·' for b in d[:mx]))

    @staticmethod
    def _is_garbled(text, min_word_len=3):
        """Check if decrypted preview is actually readable vs garbled junk."""
        if not text or len(text) < 10: return True
        # Count dots (our replacement for non-printable bytes)
        dot_ratio = text.count('·') / len(text)
        if dot_ratio > 0.3: return True
        # Check character diversity
        unique = len(set(text.replace('·','')))
        if unique < 10 and len(text) > 50: return True
        # Check for actual words (3+ letter sequences)
        words = re.findall(r'[a-zA-Z]{3,}', text)
        if len(text) > 50 and len(words) < 2: return True
        return False

    @classmethod
    def attempt_all(cls, data, ml, known_plaintext=None):
        """Run decryption methods appropriate to the data type."""
        proofs=[]; tr=" ".join(ml.get("triggers",[])).lower()
        fe=ml.get("features",{})
        ent=fe.get("shannon_entropy",8); asc=fe.get("ascii_ratio",0)

        # ── Protocol detection — skip most methods on proper encryption ──
        proto = ProtocolDetector.detect(data)
        is_tls = proto.get("protocol") == "TLS"
        is_proper_crypto = proto.get("encrypted_properly", False)
        is_binary = ProtocolDetector.is_binary_noise(data)

        # For properly encrypted protocol traffic, only check for:
        # - PAN leakage (should never happen but critical if it does)
        # - String extraction (leaked plaintext in encrypted stream)
        # - Known-plaintext attack (if user provides a crib, honor it)
        if is_proper_crypto:
            log.debug(f"Proper {proto.get('protocol','')} detected — limited analysis")
            pt = cls._strings(data)
            # Only report strings if they contain actual sensitive data
            if pt and pt.get("sensitive"):
                proofs.append(pt)
            if known_plaintext and len(known_plaintext) >= 4:
                kp = cls._known_plaintext(data, known_plaintext)
                if kp and kp["success"]: proofs.append(kp)
            # Report protocol info as a finding
            proofs.append({"method":"Protocol Detection","success":True,
                "protocol":proto.get("protocol"),
                "version":proto.get("version",""),
                "weak_version":proto.get("weak_version",False),
                "confidence":0.9,"printable":0.0,
                "note":"Properly encrypted protocol traffic detected"})
            return proofs

        # ── For non-protocol data, run appropriate methods ──

        # String extraction — always try
        pt = cls._strings(data)
        if pt: proofs.append(pt)

        # Skip text-based attacks on high-entropy binary data
        if is_binary and ent > 7.0:
            # XOR is still valid on binary if IoC suggests repeating key
            if fe.get("xor_ioc_ratio",0) > 2.5:
                x = cls._xor(data)
                # XOR partial recovery is still useful — keep if >40% printable
                if x and x["success"] and x.get("printable",0) > 0.4: proofs.append(x)
            rp = cls._repeating_pattern(data)
            if rp and rp["success"]: proofs.append(rp)
            if known_plaintext and len(known_plaintext) >= 4:
                kp = cls._known_plaintext(data, known_plaintext)
                if kp and kp["success"]: proofs.append(kp)
            if fe.get("block16_dup",0) > 0.01:
                e = cls._ecb(data)
                if e: proofs.append(e)
            log.debug(f"High-entropy binary — limited to XOR/pattern/ECB/crib")
            return proofs

        # Multi-layer peel — only if data is mostly printable (actual encoding)
        if asc > 0.8:
            peel = cls._peel_layers(data)
            if peel and peel["success"] and not cls._is_garbled(peel.get("final_preview","")):
                proofs.append(peel)

        # XOR
        if fe.get("xor_ioc_ratio",0) > 2.5 or "xor" in tr or (ent < 7.5 and not is_binary):
            x = cls._xor(data)
            if x and x["success"] and x.get("printable",0) > 0.4: proofs.append(x)

        # Base64 / hex decode — only on printable data
        if asc > 0.75:
            b = cls._b64(data)
            if b and b["success"]: proofs.append(b)
            h = cls._hex_decode(data)
            if h and h["success"]: proofs.append(h)

        # Caesar — always try (fast), validate output
        c = cls._caesar(data)
        if c and c["success"]: proofs.append(c)

        # ROT13 / ROT47
        r13 = cls._rot(data)
        if r13 and r13["success"]: proofs.append(r13)

        # ECB
        if fe.get("block16_dup",0) > 0.01 or "ecb" in tr:
            e = cls._ecb(data)
            if e: proofs.append(e)

        # Vigenère — only on letter-heavy data, high threshold
        letter_count = sum(1 for b in data[:1024] if (65<=b<=90) or (97<=b<=122))
        if letter_count > 50:
            v = cls._vigenere(data)
            if v and v["success"]: proofs.append(v)

        # Repeating pattern
        rp = cls._repeating_pattern(data)
        if rp and rp["success"]: proofs.append(rp)

        # Known plaintext
        if known_plaintext and len(known_plaintext) >= 4:
            kp = cls._known_plaintext(data, known_plaintext)
            if kp and kp["success"]: proofs.append(kp)

        # URL decode — only if meaningful %XX sequences
        if b'%' in data and asc > 0.7:
            ud = cls._url_decode(data)
            if ud and ud["success"]: proofs.append(ud)

        # Entropy windowing — only report if it finds actual mixed content
        if len(data) > 200:
            ew = cls._entropy_window(data)
            if ew and ew["success"] and ew.get("mixed_content"):
                proofs.append(ew)

        # RC4 bias detection — on high-entropy non-protocol data
        if ent > 7.0 and not is_binary:
            rc = cls._rc4_bias(data)
            if rc and rc["success"]: proofs.append(rc)

        # Byte substitution analysis — on data with letter-like frequency distribution
        if not is_binary and 3.5 < ent < 7.5:
            sub = cls._byte_frequency(data)
            if sub and sub["success"]: proofs.append(sub)

        log.debug(f"Decryption: {len(proofs)} proofs from {len(data)} bytes (proto={proto.get('protocol','none')})")
        return proofs

    @classmethod
    def _xor(cls, data):
        d=data[:4096]; n=len(d)
        if n<32: return None
        ri=1.0/256; bk=1; bi=0
        for kl in range(1,min(33,n//4)):
            st=[[] for _ in range(kl)]
            for i,b in enumerate(d): st[i%kl].append(b)
            iv=[sum(c*(c-1) for c in collections.Counter(s).values())/(len(s)*(len(s)-1))
                for s in st if len(s)>=2]
            ai=np.mean(iv) if iv else 0
            if ai>bi: bi=ai; bk=kl
        if bi<ri*2.5: return None
        key=bytearray(bk)
        for pos in range(bk):
            stream=[d[i] for i in range(pos,n,bk)]
            top=collections.Counter(stream).most_common(1)[0][0]
            bs=0; bkb=0
            for cp in [0x20,0x00,0x65,0x74,0x61,0x6F,0x0A,0x0D,0x30,0x2C]:
                ck=top^cp; sc=cls._pr(bytes(b^ck for b in stream))
                if sc>bs: bs=sc; bkb=ck
            key[pos]=bkb
        dec=bytes(d[i]^key[i%bk] for i in range(n))
        ps=cls._pr(dec); ss=cls._st(dec)
        if ps<0.3 and ss<0.1: return None
        return {"method":"XOR Key Recovery","success":True,"confidence":round(max(ps,ss),3),
            "key_hex":key.hex(),"key_len":bk,"preview":cls._prev(dec),
            "printable":round(ps,3),"structure":round(ss,3),
            "key_ascii":''.join(chr(b) if 32<=b<=126 else f'\\x{b:02x}' for b in key)}

    @classmethod
    def _caesar(cls, data):
        """Brute force all 256 byte shifts. Ranks by word matches + natural text quality."""
        d=data[:2048]
        common_words={b'the',b'and',b'for',b'are',b'with',b'this',b'that',b'have',
                      b'from',b'card',b'name',b'http',b'post',b'over',b'they',
                      b'been',b'will',b'your',b'what',b'data',b'quick',b'brown',
                      b'fox',b'jumps',b'lazy',b'dog',b'password',b'amount'}
        candidates=[]
        for sh in range(1,256):
            dec=bytes((b-sh)%256 for b in d)
            pr=cls._pr(dec)
            if pr<0.5: continue
            words=re.findall(rb'[a-zA-Z]{3,}',dec)
            wm=sum(1 for w in words if w.lower() in common_words)
            if wm<1: continue
            # Quality scoring beyond just word matches:
            null_count=dec.count(0)
            null_penalty=null_count*0.1  # Real text doesn't have null bytes
            letters=[b for b in dec if (65<=b<=90) or (97<=b<=122)]
            if letters:
                lower_ratio=sum(1 for b in letters if 97<=b<=122)/len(letters)
            else:
                lower_ratio=0
            # Natural English is ~95% lowercase. All-uppercase is suspicious.
            case_bonus=2.0 if 0.5<lower_ratio<0.99 else 0
            sc=wm+case_bonus-null_penalty
            candidates.append((sh,dec,pr,wm,sc))
        if not candidates: return None
        candidates.sort(key=lambda x:(-x[4],-x[3],-x[2]))
        sh,bd,pr,wm,sc=candidates[0]
        return {"method":"Caesar/Shift","success":True,"confidence":round(min((pr+wm*0.3)/2,1),3),
            "shift":sh,"preview":cls._prev(bd),"printable":round(pr,3),"word_hits":wm}

    @classmethod
    def _rot(cls, data):
        """Try ROT13 (letters only) and ROT47 (printable ASCII)."""
        d=data[:2048]
        # ROT13
        r13=bytearray(len(d))
        for i,b in enumerate(d):
            if 65<=b<=90: r13[i]=(b-65+13)%26+65
            elif 97<=b<=122: r13[i]=(b-97+13)%26+97
            else: r13[i]=b
        r13=bytes(r13)
        # ROT47
        r47=bytearray(len(d))
        for i,b in enumerate(d):
            if 33<=b<=126: r47[i]=(b-33+47)%94+33
            else: r47[i]=b
        r47=bytes(r47)
        # Score by English word hits, not just printability
        common={b'the',b'and',b'for',b'are',b'with',b'this',b'that',b'have',b'from',
                b'card',b'name',b'http',b'post',b'password',b'amount',b'number',
                b'over',b'they',b'been',b'will',b'your',b'what',b'when',b'make'}
        def word_score(text):
            words=re.findall(rb'[a-zA-Z]{3,}',text)
            return sum(1 for w in words if w.lower() in common)
        orig_w=word_score(d); r13_w=word_score(r13); r47_w=word_score(r47)
        if r13_w>orig_w+1 and r13_w>=2:
            return {"method":"ROT13","success":True,"confidence":round(min(r13_w/5,1),3),
                "preview":cls._prev(r13),"printable":round(cls._pr(r13),3),"word_hits":r13_w}
        if r47_w>orig_w+1 and r47_w>=2:
            return {"method":"ROT47","success":True,"confidence":round(min(r47_w/5,1),3),
                "preview":cls._prev(r47),"printable":round(cls._pr(r47),3),"word_hits":r47_w}
        return None

    @classmethod
    def _vigenere(cls, data):
        """Crack Vigenère cipher via IoC key-length detection + frequency analysis."""
        d=data[:4096]
        letters=bytes(b for b in d if (65<=b<=90) or (97<=b<=122))
        if len(letters)<40: return None
        upper=bytes(b if 65<=b<=90 else b-32 for b in letters)
        n=len(upper)

        # IoC-based key length detection (more reliable than Kasiski for shorter texts)
        english_ioc=0.065; random_ioc=1.0/26
        best_kl=0; best_ioc=0
        for kl in range(2, min(21, n//5)):
            streams=[upper[pos::kl] for pos in range(kl)]
            iocs=[]
            for stream in streams:
                if len(stream)<5: continue
                freq=collections.Counter(stream); sn=len(stream)
                ic=sum(c*(c-1) for c in freq.values())/(sn*(sn-1)) if sn>1 else 0
                iocs.append(ic)
            if iocs:
                avg=np.mean(iocs)
                if avg>best_ioc: best_ioc=avg; best_kl=kl

        if best_ioc<random_ioc*1.3 or best_kl==0: return None

        # Recover key position by position using chi-squared against English freq
        english_freq=[0.082,0.015,0.028,0.043,0.127,0.022,0.020,0.061,0.070,0.002,
                      0.008,0.040,0.024,0.067,0.075,0.019,0.001,0.060,0.063,0.091,
                      0.028,0.010,0.023,0.002,0.020,0.001]  # A-Z frequencies

        best_key=bytearray(best_kl)
        for pos in range(best_kl):
            stream=upper[pos::best_kl]
            if len(stream)<3: continue
            freq=collections.Counter(stream)
            sn=len(stream)
            best_chi=float('inf'); best_shift=0
            for shift in range(26):
                chi=0
                for letter in range(26):
                    observed=freq.get(((letter+shift)%26)+65,0)/sn
                    expected=english_freq[letter]
                    chi+=(observed-expected)**2/(expected+1e-10)
                if chi<best_chi:
                    best_chi=chi; best_shift=shift
            best_key[pos]=best_shift

        # Decrypt and score
        dec=bytearray(n)
        for i,b in enumerate(upper):
            dec[i]=(b-65-best_key[i%best_kl])%26+65
        text=bytes(dec).decode('ascii',errors='ignore').lower()
        best_score=0
        for w in ['the','and','that','have','for','are','with','this','from','they',
                   'been','said','each','not','will','all','can','had','her','was',
                   'one','our','out','you','day','get','has','him','his','how',
                   'man','new','now','old','see','way','who','did','let','say',
                   'she','too','use','need','attack','dawn','secret','force','great',
                   'ready','expect','assault','card','number','name','able','about',
                   'after','also','back','because','but','come','could','first',
                   'give','good','into','just','know','like','look','make','most',
                   'much','must','only','other','over','people','right','some',
                   'take','than','them','then','there','these','think','time',
                   'very','want','well','what','when','which','work','would','year']:
            best_score+=text.count(w)

        if best_score<5: return None

        # Full decrypt preserving case and non-letters
        dec_full=bytearray(len(d)); ki=0
        for i,b in enumerate(d):
            if 65<=b<=90:
                dec_full[i]=(b-65-best_key[ki%best_kl])%26+65; ki+=1
            elif 97<=b<=122:
                dec_full[i]=(b-97-best_key[ki%best_kl])%26+97; ki+=1
            else:
                dec_full[i]=b

        key_letters=''.join(chr(k+65) for k in best_key)
        return {"method":"Vigenère Crack","success":True,
            "confidence":round(min(best_score/8,1.0),3),
            "key_letters":key_letters,"key_len":best_kl,"word_hits":best_score,
            "preview":cls._prev(bytes(dec_full)),
            "printable":round(cls._pr(bytes(dec_full)),3)}

    @classmethod
    def _repeating_pattern(cls, data):
        """Find any repeating byte patterns (custom ciphers, padding, etc.)."""
        d=data[:4096]; n=len(d)
        if n<32: return None
        results=[]
        # Check for repeating sequences of length 2-64
        for plen in [2,3,4,5,6,7,8,12,16,24,32,48,64]:
            if n<plen*3: continue
            pattern=d[:plen]
            matches=0
            for i in range(0,n-plen+1,plen):
                if d[i:i+plen]==pattern: matches+=1
            ratio=matches/(n//plen)
            if ratio>0.5:
                results.append({"length":plen,"matches":matches,"total":n//plen,
                    "ratio":round(ratio,3),
                    "hex":pattern.hex()[:32],
                    "ascii":''.join(chr(b) if 32<=b<=126 else '·' for b in pattern[:16])})
        # Also check XOR pattern against itself (self-correlation peaks)
        autocorr=[]
        arr=np.frombuffer(d[:1024],dtype=np.uint8).astype(float)
        for lag in range(2,min(65,n//3)):
            if lag>=len(arr): break
            corr=np.corrcoef(arr[:-lag],arr[lag:])[0,1]
            if not np.isnan(corr) and corr>0.5:
                autocorr.append({"lag":lag,"correlation":round(float(corr),4)})
        if not results and not autocorr: return None
        return {"method":"Pattern Detection","success":True,
            "repeating_patterns":results[:5],"autocorrelation_peaks":autocorr[:5],
            "confidence":round(max((r["ratio"] for r in results),default=0),3) if results else 0.3,
            "printable":0.0}

    @classmethod
    def _known_plaintext(cls, data, known):
        """Known-plaintext (crib) attack: find short repeating XOR keys."""
        d=data[:8192]; kn=known[:256]
        if len(d)<len(kn) or len(kn)<4: return None
        results=[]
        MAX_KEY_LEN=16  # Real XOR keys are short — 18-31 byte "keys" are noise
        for offset in range(min(len(d)-len(kn)+1, 500)):
            segment=d[offset:offset+len(kn)]
            key_candidate=bytes(a^b for a,b in zip(segment,kn))
            for kl in range(1,min(MAX_KEY_LEN+1,len(key_candidate))):
                key_test=key_candidate[:kl]
                matches=sum(1 for i in range(len(key_candidate))
                            if key_candidate[i]==key_test[i%kl])
                if matches==len(key_candidate):
                    dec=bytes(d[i]^key_test[i%kl] for i in range(len(d)))
                    ps=cls._pr(dec); ss=cls._st(dec)
                    if ps>0.6 or ss>0.2:
                        preview=cls._prev(dec)
                        if not cls._is_garbled(preview):
                            results.append({
                                "offset":offset,"key_hex":key_test.hex(),
                                "key_len":kl,"printable":round(ps,3),
                                "structure":round(ss,3),
                                "preview":preview,
                                "key_ascii":''.join(chr(b) if 32<=b<=126 else f'\\x{b:02x}' for b in key_test)
                            })
                    break
        if not results: return None
        best=max(results,key=lambda r:r["printable"]+r["structure"])
        return {"method":"Known-Plaintext Attack","success":True,
            "confidence":round(max(best["printable"],best["structure"]),3),
            "key_hex":best["key_hex"],"key_len":best["key_len"],
            "key_ascii":best.get("key_ascii",""),
            "offset":best["offset"],"preview":best["preview"],
            "printable":best["printable"],"structure":best["structure"],
            "candidates_found":len(results)}

    @classmethod
    def _peel_layers(cls, data):
        """Recursively peel encoding layers. Tries hex before base64 (hex is subset of b64)."""
        d=data; layers=[]; max_layers=5
        for _ in range(max_layers):
            peeled=False
            input_printable=cls._pr(d)
            if input_printable < 0.7: break  # Not printable enough to be encoded
            # Try hex FIRST (hex chars are a subset of base64 chars)
            try:
                clean=d.strip().decode('ascii',errors='ignore')
                hex_clean=re.sub(r'[\s:.-]','',clean)
                if len(hex_clean)>=16 and len(hex_clean)%2==0 and all(c in '0123456789abcdefABCDEF' for c in hex_clean):
                    dec=bytes.fromhex(hex_clean)
                    if len(dec)>=8 and dec!=d:
                        layers.append(("hex",len(d),len(dec)))
                        d=dec; peeled=True; continue
            except: pass
            # Try base64
            try:
                clean=d.strip()
                b64c=set(b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r')
                if len(clean)>16 and sum(1 for b in clean if b in b64c)/len(clean)>0.9:
                    dec=base64.b64decode(clean,validate=True)
                    if len(dec)>=8 and dec!=d:
                        layers.append(("base64",len(d),len(dec)))
                        d=dec; peeled=True; continue
            except: pass
            # Try URL decode
            try:
                text=d.decode('ascii',errors='ignore')
                valid_pct=len(re.findall(r'%[0-9A-Fa-f]{2}',text))
                if valid_pct >= 5:
                    from urllib.parse import unquote_to_bytes
                    dec=unquote_to_bytes(text)
                    if dec!=d and len(dec)>=8 and len(dec)<len(d):
                        layers.append(("url",len(d),len(dec)))
                        d=dec; peeled=True; continue
            except: pass
            break

        if len(layers)<2: return None
        ps=cls._pr(d); ss=cls._st(d)
        if ps<0.4 and ss<0.1: return None
        if cls._is_garbled(cls._prev(d)): return None
        pan=PANDetector.scan(d)
        return {"method":"Multi-Layer Decode","success":True,
            "layers":[{"encoding":e,"input_size":i,"output_size":o} for e,i,o in layers],
            "total_layers":len(layers),
            "final_preview":cls._prev(d),"printable":round(ps,3),"structure":round(ss,3),
            "card_data":len(pan)>0,
            "confidence":round(max(ps,ss),3)}

    @classmethod
    def _hex_decode(cls, data):
        """Try hex decoding."""
        try:
            clean=data.strip().decode('ascii',errors='ignore')
            # Remove 0x prefixes first (as whole token), then formatting chars
            hex_clean=re.sub(r'0[xX]','',clean)  # Remove 0x prefix
            hex_clean=re.sub(r'[\s:.\-]','',hex_clean)  # Remove separators (NOT 0 or x!)
            if len(hex_clean)<8 or len(hex_clean)%2!=0: return None
            if not all(c in '0123456789abcdefABCDEF' for c in hex_clean): return None
            dec=bytes.fromhex(hex_clean)
            ps=cls._pr(dec); ss=cls._st(dec)
            if ps<0.3 and ss<0.1: return None
            return {"method":"Hex Decode","success":True,"confidence":round(max(ps,ss),3),
                "preview":cls._prev(dec),"decoded_size":len(dec),"printable":round(ps,3)}
        except: return None

    @classmethod
    def _url_decode(cls, data):
        """URL decode — validates that decoding reveals meaningful content."""
        try:
            from urllib.parse import unquote_to_bytes
            text=data[:4096].decode('ascii',errors='ignore')
            valid_pct=len(re.findall(r'%[0-9A-Fa-f]{2}',text))
            if valid_pct < 3: return None
            dec=unquote_to_bytes(text)
            if dec==data[:4096] or dec==text.encode(): return None
            ps=cls._pr(dec); ss=cls._st(dec)
            # Decoded must be shorter (encoding removed) and structured
            if len(dec)>=len(data[:4096]): return None
            if ps<0.4 and ss<0.1: return None
            # Must have meaningful structure OR high readability
            if ss<0.08 and valid_pct<5: return None
            return {"method":"URL Decode","success":True,"confidence":round(max(ps,ss),3),
                "preview":cls._prev(dec),"printable":round(ps,3),"pct_sequences":valid_pct}
        except: return None

    @classmethod
    def _entropy_window(cls, data, window=None):
        """Sliding window entropy — detect mixed encrypted/plaintext sections."""
        d=data[:8192]; n=len(d)
        if n<100: return None
        if window is None:
            window=min(128, max(32, n//6))
        if n<window*3: window=max(16, n//4)
        step=max(8, window//2)
        windows=[]
        for i in range(0,n-window+1,step):
            chunk=d[i:i+window]; cl=len(chunk)
            if cl<8: continue
            freq=collections.Counter(chunk)
            ent=-sum((c/cl)*math.log2(c/cl) for c in freq.values())
            windows.append({"offset":i,"entropy":round(ent,3),"size":cl})
        if len(windows)<3: return None
        ents=[w["entropy"] for w in windows]
        ent_range=max(ents)-min(ents)
        # Use adaptive thresholds: split at the median entropy
        median_ent=float(np.median(ents))
        low=[w for w in windows if w["entropy"]<min(5.0, median_ent-0.5)]
        high=[w for w in windows if w["entropy"]>max(median_ent+0.5, 5.5)]
        mixed=len(low)>0 and len(high)>0 and ent_range>1.5
        if not mixed and not low: return None
        transitions=[]
        for i in range(1,len(windows)):
            diff=abs(windows[i]["entropy"]-windows[i-1]["entropy"])
            if diff>1.0:
                transitions.append({"offset":windows[i]["offset"],"from":windows[i-1]["entropy"],
                    "to":windows[i]["entropy"],"type":"encrypted→plain" if windows[i]["entropy"]<windows[i-1]["entropy"] else "plain→encrypted"})
        return {"method":"Entropy Windowing","success":True,
            "min_entropy":round(min(ents),3),"max_entropy":round(max(ents),3),
            "avg_entropy":round(np.mean(ents),3),
            "low_entropy_sections":len(low),"high_entropy_sections":len(high),
            "mixed_content":mixed,"transitions":transitions[:10],
            "plaintext_regions":[{"offset":w["offset"],"entropy":w["entropy"],
                "preview":cls._prev(d[w["offset"]:w["offset"]+window],80)} for w in low[:5]],
            "confidence":0.5 if mixed else 0.2,
            "printable":0.0}

    @classmethod
    def _rc4_bias(cls, data):
        """Detect RC4 stream cipher via statistical biases in output.
        RC4 has known biases: byte 2 has P(0x00)≈2/256, and early bytes
        show non-uniform distribution (Fluhrer-Mantin-Shamir attack indicator)."""
        if len(data) < 256: return None
        d = data[:4096]
        
        # Check byte 2 bias (0x00 appears ~2× expected in RC4 byte 2 across many streams)
        # For a single stream, check first-byte distribution skew
        first_bytes = d[:256]
        freq = collections.Counter(first_bytes)
        
        # Chi-squared test against uniform distribution
        expected = len(first_bytes) / 256
        chi_sq = sum((freq.get(i, 0) - expected) ** 2 / max(expected, 0.01) for i in range(256))
        
        # RC4 has moderate chi-squared (not uniform, not plaintext)
        # True random: chi_sq ≈ 256, RC4: chi_sq ≈ 280-400, plaintext: chi_sq > 1000
        if not (260 < chi_sq < 500): return None
        
        # Check for key scheduling artifacts — first 256 bytes should show specific patterns
        # The Mantin bias: P(Z_r = 0) > 1/256 for small r
        zero_positions = [i for i in range(min(32, len(d))) if d[i] == 0]
        zero_density = len(zero_positions) / min(32, len(d))
        
        # RC4 typically has slightly elevated zeros in first 32 bytes
        if zero_density < 0.05 or zero_density > 0.3: return None
        
        # Check entropy — RC4 output is high entropy but not maximal
        sample = d[:1024]
        freq2 = collections.Counter(sample)
        ent = -sum((c/len(sample)) * math.log2(c/len(sample)) for c in freq2.values())
        if ent < 7.0 or ent > 7.95: return None
        
        return {"method": "RC4 Bias Detection", "success": True,
                "confidence": round(min((chi_sq - 256) / 200, 0.8), 3),
                "chi_squared": round(chi_sq, 1),
                "zero_bias": round(zero_density, 3),
                "entropy": round(ent, 3),
                "printable": 0.0,
                "note": "Statistical biases suggest RC4 stream cipher"}

    @classmethod
    def _byte_frequency(cls, data):
        """Detect simple byte substitution cipher via frequency analysis.
        If the byte frequency distribution matches English letter frequency
        (just shifted to different byte values), it's a substitution cipher."""
        if len(data) < 200: return None
        d = data[:8192]
        
        # English letter frequencies (approximate, for lowercase)
        english_freq = [0.082,0.015,0.028,0.043,0.127,0.022,0.020,0.061,0.070,0.002,
                        0.008,0.040,0.024,0.067,0.075,0.019,0.001,0.060,0.063,0.091,
                        0.028,0.010,0.023,0.002,0.020,0.001]
        english_sorted = sorted(english_freq, reverse=True)
        
        # Get actual byte frequency distribution
        freq = collections.Counter(d)
        total = len(d)
        byte_freqs = [(b, c/total) for b, c in freq.most_common()]
        actual_sorted = [f for _, f in byte_freqs[:26]]
        
        if len(actual_sorted) < 20: return None
        
        # Correlation between sorted actual and sorted English
        # High correlation = substitution cipher
        min_len = min(len(actual_sorted), len(english_sorted))
        correlation = sum(a * e for a, e in zip(actual_sorted[:min_len], english_sorted[:min_len]))
        expected_corr = sum(e * e for e in english_sorted[:min_len])  # Perfect match
        random_corr = sum(1/256 * e for e in english_sorted[:min_len])  # Random
        
        if correlation < random_corr * 1.5: return None  # Too close to random
        
        score = (correlation - random_corr) / max(expected_corr - random_corr, 0.001)
        if score < 0.3: return None
        
        # Build substitution mapping guess (most frequent byte → 'e', etc.)
        common_english = 'etaoinshrdlcumwfgypbvkjxqz'
        mapping = {}
        for i, (byte_val, _) in enumerate(byte_freqs[:26]):
            mapping[byte_val] = ord(common_english[i])
        
        # Apply mapping to get preview
        dec = bytearray(len(d))
        for i, b in enumerate(d):
            dec[i] = mapping.get(b, b)
        
        preview = cls._prev(bytes(dec))
        # Frequency mapping is approximate — score is the real indicator, not readability
        if score < 0.5: return None
        
        return {"method": "Substitution Analysis", "success": True,
                "confidence": round(min(score, 1.0), 3),
                "frequency_correlation": round(score, 3),
                "unique_bytes": len(freq),
                "top_byte": f"0x{byte_freqs[0][0]:02x} ({byte_freqs[0][1]:.1%})",
                "preview": preview,
                "printable": round(cls._pr(bytes(dec)), 3),
                "note": "Byte frequency matches natural language distribution"}

    @classmethod
    def _b64(cls, data):
        try:
            cl=data.strip()
            b64c=set(b'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r')
            if sum(1 for b in cl if b in b64c)/len(cl)<0.9: return None
            dec=base64.b64decode(cl,validate=True); sc=cls._pr(dec); ss=cls._st(dec)
            if sc<0.3 and ss<0.1: return None
            return {"method":"Base64 Decode","success":True,"confidence":round(max(sc,ss),3),
                "preview":cls._prev(dec),"decoded_size":len(dec),
                "card_data":len(PANDetector.scan(dec))>0,"printable":round(sc,3)}
        except: return None

    @classmethod
    def _ecb(cls, data, bs=16):
        if len(data)<bs*4: return None
        blocks=[data[i:i+bs] for i in range(0,len(data)-bs+1,bs)]
        bc=collections.Counter(blocks)
        if len(bc)==len(blocks): return None
        top=[{"hex":bd.hex()[:32],"count":cnt,
              "ascii":''.join(chr(b) if 32<=b<=126 else '·' for b in bd),
              "pct":round(cnt/len(blocks)*100,1)} for bd,cnt in bc.most_common(5)]
        ids={}; nid=0; bmap=[]
        for b in blocks:
            if b not in ids: ids[b]=nid; nid+=1
            bmap.append(ids[b])
        chars="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        return {"method":"ECB Analysis","success":True,"total":len(blocks),"unique":len(bc),
            "ratio":round(len(bc)/len(blocks),3),"top_blocks":top,
            "map":''.join(chars[x%len(chars)] for x in bmap[:80])}

    @classmethod
    def _strings(cls, data, minl=8):
        """Extract readable strings, filtering protocol noise."""
        strs=[]; cur=[]
        for b in data[:8192]:
            if 32<=b<=126: cur.append(chr(b))
            else:
                if len(cur)>=minl: strs.append(''.join(cur))
                cur=[]
        if len(cur)>=minl: strs.append(''.join(cur))
        if not strs: return None

        # Filter out TLS/cert/protocol noise strings
        tls_noise={'.com','.org','.net','DigiCert','VeriSign','GlobalSign','Comodo',
            'Certificate','Authority','Subject','Issuer','Serial','Public Key',
            'Signature Algorithm','sha256','sha384','RSA','ECDSA','X509',
            'BEGIN CERT','END CERT','countryName','organizat','localityName',
            'stateOrProvince','commonName','http/1.1','h2','spdy','grpc'}
        filtered=[]
        for s in strs:
            is_noise=False
            for noise in tls_noise:
                if noise.lower() in s.lower(): is_noise=True; break
            # Also filter single-word strings that look like OIDs or hex
            if re.match(r'^[0-9.]+$',s): is_noise=True
            if re.match(r'^[0-9a-f]+$',s,re.IGNORECASE) and len(s)<20: is_noise=True
            if not is_noise: filtered.append(s)
        strs=filtered
        if not strs: return None

        full='\n'.join(strs); sens=[]
        # Card numbers — must pass Luhn check AND not be a timestamp
        for m in re.findall(r'\d{13,19}', full)[:5]:
            # Filter timestamps (YYYYMMDD... patterns)
            if m[:4] in ('2024','2025','2026','2027','2028','2029','2030'): continue
            if m[:2] in ('19','20') and len(m)==14: continue  # YYYYMMDDHHmmss
            if PANDetector.luhn(m):
                sens.append(("card#", re.sub(r'(\d{6})\d{6,10}(\d{4})', r'\1******\2', m)))
        # Other sensitive patterns
        for pat,lbl in [
            (r'(?i)(password|passwd|secret|pin)\s*[=:]\s*[^\s&;,]+',"cred"),
            (r'(?i)(card|pan|cvv|cvc|exp)\s*[=:_]\s*[^\s&;,]+',"card_ref"),
            (r'HTTP/\d',"HTTP"),
            (r'(GET|POST|PUT|DELETE)\s+/',"HTTP_method"),
            (r'\d+\.\d+\.\d+\.\d+',"IP"),
            (r'(?i)(bearer|token|api.?key|session)\s*[=:]\s*[^\s&;,]+',"auth_token"),
            (r'(?i)(ssn|social)\s*[=:]\s*\d{3}.?\d{2}.?\d{4}',"SSN"),
            (r'(?i)(dob|birth)\s*[=:]\s*\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}',"DOB")]:
            for m in re.findall(pat,full,re.IGNORECASE)[:3]:
                sens.append((lbl,re.sub(r'(\d{6})\d{6,10}(\d{4})',r'\1******\2',str(m))))
        # Emails — filter SSH/TLS protocol algorithm identifiers
        ssh_noise={'libssh.org','openssh.com','ietf.org'}
        for m in re.findall(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-zA-Z]{2,}', full)[:3]:
            domain = m.split('@')[-1].lower()
            if domain not in ssh_noise and len(m.split('@')[0])>=2:
                sens.append(("email", m))
        cov=sum(len(s) for s in strs)/min(len(data),8192)
        # Only report if: has sensitive data, OR significant plaintext coverage
        if not sens and cov < 0.15: return None
        return {"method":"String Extraction","success":True,"count":len(strs),
            "coverage":round(cov,3),"samples":[s[:120] for s in strs[:10]],"sensitive":sens[:15]}

class AIValidator:
    def __init__(self, url="http://localhost:11434", model="qwen2.5:14b"):
        self.url=url.rstrip("/"); self.model=model; self._avail=None
    def _check(self):
        if self._avail is not None: return self._avail
        try:
            import urllib.request
            with urllib.request.urlopen(f"{self.url}/api/tags",timeout=3) as r: self._avail=r.status==200
        except: self._avail=False
        return self._avail
    def _query(self, prompt):
        if not self._check(): return None
        try:
            import urllib.request
            d=json.dumps({"model":self.model,"prompt":prompt,"stream":False,
                "options":{"temperature":0.1,"num_predict":200}}).encode()
            req=urllib.request.Request(f"{self.url}/api/generate",data=d,
                headers={"Content-Type":"application/json"},method="POST")
            with urllib.request.urlopen(req,timeout=15) as r:
                return json.loads(r.read()).get("response","")
        except: return None
    def validate(self, proof, data, use_ai=False):
        h=self._heuristic(proof)
        ai=self._ai_val(proof) if use_ai else None
        sc=0.4*h["score"]+0.6*ai["score"] if ai else h["score"]
        vd=ai.get("verdict",h["verdict"]) if ai else h["verdict"]
        log.debug(f"Validation: {proof.get('method')}: {vd} ({sc:.0%})")
        return {"score":round(sc,3),"verdict":vd,"data_types":h["data_types"],
            "corrections":h["corrections"],"ai_available":ai is not None,
            "ai_reasoning":ai.get("reasoning","") if ai else "",
            "ml_correction":self._corr(sc,vd,proof)}
    def _heuristic(self, proof):
        text=(proof.get("preview","") or "").encode(); score=0.0; types=[]; corr=[]
        pr=proof.get("printable",0)
        if pr>0.85: score+=0.3
        elif pr>0.6: score+=0.15
        for dt,(pat,w) in {"HTTP":(rb'HTTP/|GET |POST ',0.2),"JSON":(rb'[{"\[]',0.15),
            "CARD":(rb'4[0-9]{12}|5[1-5]',0.25),"CARD_FIELD":(rb'(?i)(card|pan|cvv)',0.15),
            "KV":(rb'[a-zA-Z_]+=\S+',0.1)}.items():
            if re.search(pat,text): types.append(dt); score+=w
        words=re.findall(rb'[a-zA-Z]{3,}',text)
        cw={b'the',b'and',b'card',b'name',b'http',b'post',b'amount',b'auth',b'password'}
        if sum(1 for w in words if w.lower() in cw)>=3: score+=0.2; types.append("ENGLISH")
        prev=proof.get("preview","")
        if len(set(prev))<15 and len(prev)>50: score-=0.3; corr.append("Garbled output")
        if re.search(r'(.{2,6})\1{5,}',prev): score-=0.25; corr.append("Repetitive — wrong key")
        if proof.get("method")=="Base64 Decode" and proof.get("card_data"): score+=0.3
        if proof.get("method")=="ECB Analysis" and proof.get("ratio",1)<0.5: score+=0.2; types.append("ECB")
        score=max(0,min(1,score))
        v="CONFIRMED" if score>=0.7 else "LIKELY" if score>=0.4 else "PARTIAL" if score>=0.2 else "UNCONFIRMED"
        return {"score":round(score,3),"verdict":v,"data_types":types,"corrections":corr or(["No structure"] if score<0.2 else [])}
    def _ai_val(self, proof):
        prev=proof.get("preview","")
        if not prev or len(prev)<10: return None
        resp=self._query(f"Analyze this decryption. Is it real plaintext or garbled? "
            f"Method: {proof.get('method','')}. Text: {prev[:200]}. "
            f"Respond JSON: {{\"score\":0-1,\"verdict\":\"CONFIRMED|LIKELY|PARTIAL|UNCONFIRMED\",\"reasoning\":\"...\"}}")
        if not resp: return None
        try:
            m=re.search(r'\{[^{}]*\}',resp,re.DOTALL)
            if m: return json.loads(m.group())
        except: pass
        return None
    def _corr(self, score, verdict, proof):
        if verdict in ("CONFIRMED",): return {"should_correct":True,"new_label":2,"reason":f"Confirmed via {proof.get('method','')}"}
        if verdict=="LIKELY": return {"should_correct":True,"new_label":2,"reason":f"Likely break via {proof.get('method','')}"}
        if verdict=="UNCONFIRMED": return {"should_correct":True,"new_label":1,"reason":"Garbled — may be stronger"}
        return {"should_correct":False,"reason":"","new_label":None}

    def analyze_traffic(self, stream_id, data, ml_result, proofs, proto):
        """Use AI to reason about what this traffic IS and suggest approaches.
        Called on interesting non-TLS streams to get deeper analysis."""
        if not self._check(): return None
        # Build context for the AI
        sample_hex = data[:64].hex()
        sample_ascii = ''.join(chr(b) if 32<=b<=126 else '.' for b in data[:128])
        fe = ml_result.get("features",{})
        proof_summary = "; ".join(f"{p['method']}(conf={p.get('confidence',0):.0%})" 
            for p in proofs if p.get("success"))

        prompt = (
            f"Analyze this network stream as a security researcher.\n"
            f"Stream: {stream_id}\n"
            f"Size: {len(data)} bytes | Entropy: {fe.get('shannon_entropy',0):.2f} | "
            f"ASCII ratio: {fe.get('ascii_ratio',0):.2f}\n"
            f"Protocol detected: {proto.get('protocol','none')}\n"
            f"ML classification: {ml_result.get('label','?')}\n"
            f"Decryption proofs: {proof_summary or 'none'}\n"
            f"First 64 bytes (hex): {sample_hex}\n"
            f"First 128 bytes (ASCII): {sample_ascii}\n\n"
            f"Respond ONLY as JSON: {{\"protocol_guess\":\"...\",\"encryption_type\":\"...\","
            f"\"risk_level\":\"high|medium|low|none\","
            f"\"findings\":\"...\",\"suggested_approach\":\"...\"}}"
        )
        resp = self._query(prompt)
        if not resp: return None
        try:
            m = re.search(r'\{[^{}]*\}', resp, re.DOTALL)
            if m:
                result = json.loads(m.group())
                result["method"] = "AI Traffic Analysis"
                result["success"] = True
                result["confidence"] = 0.5
                result["printable"] = 0.0
                return result
        except: pass
        return None

class FeedbackCollector:
    def __init__(self):
        self.session=[]; self.stats={}
        sp=FEEDBACK_DIR/"method_stats.json"
        if sp.exists():
            try: self.stats=json.loads(sp.read_text())
            except: pass
    def record(self, data, ml, proof, val):
        entry={"time":datetime.now().isoformat(),"size":len(data),
            "ml_label":ml.get("label"),"method":proof.get("method"),
            "val_score":val.get("score"),"verdict":val.get("verdict"),
            "correction":val.get("ml_correction",{})}
        self.session.append(entry)
        m=proof.get("method","?")
        if m not in self.stats:
            self.stats[m]={"attempts":0,"confirmed":0,"likely":0,"unconfirmed":0,"fp":0,"total_sc":0}
        s=self.stats[m]; s["attempts"]=s.get("attempts",0)+1
        s["total_sc"]=s.get("total_sc",0)+val.get("score",0)
        v=val.get("verdict","")
        if "CONFIRMED" in v: s["confirmed"]=s.get("confirmed",0)+1
        elif "LIKELY" in v: s["likely"]=s.get("likely",0)+1
        elif "FALSE" in v: s["fp"]=s.get("fp",0)+1
        else: s["unconfirmed"]=s.get("unconfirmed",0)+1
        (FEEDBACK_DIR/"method_stats.json").write_text(json.dumps(self.stats,indent=2))
        lid=entry["correction"].get("new_label",ml.get("prediction",0)) or 0
        ln={0:"strong",1:"moderate",2:"weak",3:"critical"}.get(lid,"unknown")
        ld=SAMPLES_DIR/ln; ld.mkdir(parents=True,exist_ok=True)
        ts=datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        (ld/f"{ts}.bin").write_bytes(data[:8192])
        log.debug(f"Feedback: {m} → {v} ({val.get('score',0):.0%})")

# ═══════════════════════════════════════════════════════════════
#  Key Harvester — cross-stream key reuse attack
# ═══════════════════════════════════════════════════════════════
class KeyHarvester:
    """Persistent key vault. Collects recovered keys across sessions,
    saves to disk, and tries them against uncracked streams.
    Keys survive app restarts so you can identify apps reusing keys
    across different captures taken days or weeks apart."""

    VAULT_FILE = FEEDBACK_DIR / "key_vault.json"

    def __init__(self):
        self.keys = []          # {"key_hex":str, "method":str, "source":str, ...}
        self.reuse_groups = {}  # key_hex → [{"stream":str, "capture":str, "time":str}]
        self._load()

    def _load(self):
        """Load key vault from disk."""
        if self.VAULT_FILE.exists():
            try:
                data = json.loads(self.VAULT_FILE.read_text())
                self.keys = data.get("keys", [])
                self.reuse_groups = data.get("reuse_groups", {})
                # Rebuild key bytes from hex
                for entry in self.keys:
                    if "key" not in entry or not isinstance(entry.get("key"), bytes):
                        try: entry["key"] = bytes.fromhex(entry["key_hex"])
                        except: entry["key"] = b''
                log.info(f"KeyVault loaded: {len(self.keys)} keys")
            except Exception as e:
                log.warning(f"KeyVault load failed: {e}")
                self.keys = []; self.reuse_groups = {}

    def _save(self):
        """Persist key vault to disk."""
        try:
            save_data = {
                "keys": [{k: v for k, v in entry.items() if k != "key"}  # Skip raw bytes
                          for entry in self.keys],
                "reuse_groups": self.reuse_groups,
                "last_updated": datetime.now().isoformat(),
                "total_keys": len(self.keys),
            }
            self.VAULT_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.VAULT_FILE.write_text(json.dumps(save_data, indent=2))
        except Exception as e:
            log.error(f"KeyVault save failed: {e}")

    def harvest(self, stream_id: str, proofs: list, capture_name: str = ""):
        """Extract keys from successful decryption proofs."""
        added = 0
        for p in proofs:
            if not p.get("success"): continue
            method = p.get("method","")
            key_bytes = None

            if method == "XOR Key Recovery" and "key_hex" in p:
                try: key_bytes = bytes.fromhex(p["key_hex"])
                except: pass
            elif method == "Known-Plaintext Attack" and "key_hex" in p:
                try: key_bytes = bytes.fromhex(p["key_hex"])
                except: pass
            elif method == "Caesar/Shift" and "shift" in p:
                key_bytes = bytes([p["shift"]])
            elif method == "Vigenère Crack" and "key_letters" in p:
                key_bytes = bytes(ord(c)-65 for c in p["key_letters"])
            elif method == "Cross-Stream Key Reuse" and "key_hex" in p:
                return  # Don't re-harvest cross-stream hits

            if key_bytes and len(key_bytes) >= 1:
                conf = p.get("confidence", 0)
                pr = p.get("printable", 0)
                # Cap key length based on method — XOR often finds multiples of true key
                max_len = 32 if method in ("XOR Key Recovery", "Vigenère Crack") else 16
                if (conf > 0.3 or pr > 0.5) and len(key_bytes) <= max_len:
                    kh = key_bytes.hex()
                    # Check for duplicate key from same source
                    if any(e["key_hex"] == kh and e["source"] == stream_id for e in self.keys):
                        continue

                    entry = {"key": key_bytes, "key_hex": kh,
                             "method": method, "source": stream_id,
                             "capture": capture_name,
                             "confidence": conf, "printable": pr,
                             "key_len": len(key_bytes),
                             "harvested_at": datetime.now().isoformat()}
                    self.keys.append(entry)
                    added += 1

                    # Track reuse — include capture context for cross-capture detection
                    sighting = {"stream": stream_id, "capture": capture_name,
                                "time": datetime.now().isoformat()}
                    self.reuse_groups.setdefault(kh, []).append(sighting)
                    log.debug(f"KeyHarvest: {method} key={kh[:16]}... from {stream_id}")

        if added > 0:
            self._save()

    def attack_stream(self, stream_id: str, data: bytes) -> list:
        """Try all harvested keys against an uncracked stream.
        Also tries sub-keys (factors of key length) since XOR crackers
        often find multiples of the true key period."""
        if not self.keys or len(data) < 16:
            return []
        findings = []
        d = data[:4096]
        best_hit = None

        for entry in self.keys:
            key = entry.get("key") or bytes.fromhex(entry["key_hex"])
            kl = len(key)
            method = entry["method"]
            source = entry["source"]

            if source == stream_id:
                continue

            if method in ("XOR Key Recovery", "Known-Plaintext Attack"):
                keys_to_try = [(key, kl)]
                for sub_len in range(1, min(kl, 33)):
                    if kl % sub_len == 0:
                        sub_key = key[:sub_len]
                        if all(key[i] == sub_key[i % sub_len] for i in range(kl)):
                            keys_to_try.append((sub_key, sub_len))
                    elif sub_len <= 16:
                        keys_to_try.append((key[:sub_len], sub_len))

                for try_key, try_len in keys_to_try:
                    dec = bytes(d[i] ^ try_key[i % try_len] for i in range(len(d)))
                    pr = DecryptionEngine._pr(dec)
                    st = DecryptionEngine._st(dec)
                    score = pr + st
                    if (pr > 0.5 or st > 0.15) and (not best_hit or score > best_hit["_score"]):
                        preview = DecryptionEngine._prev(dec)
                        if not DecryptionEngine._is_garbled(preview):
                            best_hit = {
                                "method": "Cross-Stream Key Reuse",
                                "success": True,
                                "confidence": round(max(pr, st), 3),
                                "printable": round(pr, 3),
                                "structure": round(st, 3),
                                "preview": preview,
                                "key_hex": try_key.hex(),
                                "key_len": try_len,
                                "key_ascii": ''.join(chr(b) if 32<=b<=126 else f'\\x{b:02x}' for b in try_key),
                                "source_stream": source,
                                "source_capture": entry.get("capture",""),
                                "source_method": method,
                                "source_confidence": entry["confidence"],
                                "original_key_len": kl,
                                "_score": score,
                            }

            elif method == "Caesar/Shift" and kl == 1:
                shift = key[0]
                dec = bytes((b - shift) % 256 for b in d)
                pr = DecryptionEngine._pr(dec)
                if pr > 0.6:
                    preview = DecryptionEngine._prev(dec)
                    if not DecryptionEngine._is_garbled(preview):
                        findings.append({
                            "method": "Cross-Stream Key Reuse",
                            "success": True,
                            "confidence": round(pr, 3),
                            "printable": round(pr, 3),
                            "preview": preview,
                            "key_hex": entry["key_hex"],
                            "key_len": 1,
                            "shift": shift,
                            "source_stream": source,
                            "source_capture": entry.get("capture",""),
                            "source_method": "Caesar/Shift",
                        })

        if best_hit:
            del best_hit["_score"]
            findings.append(best_hit)
        return findings

    def get_reuse_report(self) -> dict:
        """Summarize key reuse patterns, including prefix clustering and cross-capture."""
        reused = {k: sightings for k, sightings in self.reuse_groups.items() if len(sightings) > 1}
        unique_keys = len(self.reuse_groups)
        total_harvested = len(self.keys)

        # Detect cross-capture reuse
        cross_capture = {}
        for kh, sightings in reused.items():
            captures = set(s.get("capture","") for s in sightings if s.get("capture"))
            if len(captures) > 1:
                cross_capture[kh] = {"captures": list(captures), "sightings": len(sightings)}

        # 4-byte prefix clustering — keys like 13171d0c... appearing 13x across Azure
        prefix_clusters = {}
        for entry in self.keys:
            kh = entry["key_hex"]
            if len(kh) >= 8:  # Need at least 4 bytes
                prefix = kh[:8]
                prefix_clusters.setdefault(prefix, []).append(entry)
        # Only report clusters with 2+ keys
        significant_clusters = {}
        for prefix, entries in prefix_clusters.items():
            if len(entries) >= 2:
                streams = list(set(e["source"] for e in entries))
                captures = list(set(e.get("capture","") for e in entries if e.get("capture")))
                significant_clusters[prefix] = {
                    "count": len(entries),
                    "streams": streams[:10],
                    "captures": captures,
                    "full_keys": list(set(e["key_hex"] for e in entries))[:5],
                    "cross_capture": len(captures) > 1,
                }

        # Subset/superset family grouping
        key_families = {}
        for entry in self.keys:
            kh = entry["key_hex"]
            matched = False
            for fam_key in list(key_families.keys()):
                if kh in fam_key or fam_key in kh:
                    key_families[fam_key].append(entry)
                    matched = True
                    break
            if not matched:
                key_families[kh] = [entry]
        families_with_reuse = {k: v for k, v in key_families.items() if len(v) > 1}

        return {
            "total_keys_harvested": total_harvested,
            "unique_keys": unique_keys,
            "keys_reused_across_streams": len(reused),
            "reuse_details": {k: {"streams": [s.get("stream","") for s in sl],
                                   "count": len(sl)} for k, sl in reused.items()},
            "cross_capture_reuse": cross_capture,
            "key_families": len(families_with_reuse),
            "prefix_clusters": significant_clusters,
        }

    def clear(self):
        """Clear the key vault."""
        self.keys = []; self.reuse_groups = {}
        if self.VAULT_FILE.exists(): self.VAULT_FILE.unlink()
        log.info("KeyVault cleared")

# ═══════════════════════════════════════════════════════════════
#  Intel Collector — persistent credential + pattern intelligence
# ═══════════════════════════════════════════════════════════════
class IntelCollector:
    """Stores recovered credentials, plaintext↔ciphertext pairs, and
    encryption patterns across sessions. Uses stored plaintext as
    automatic cribs on future captures to detect reused encryption."""

    INTEL_FILE = FEEDBACK_DIR / "intel_vault.json"

    def __init__(self):
        self.credentials = []    # {"type":"cred/pan/token/email","value":"...","source":...}
        self.cribs = []          # {"plaintext_hex":"...","ciphertext_hex":"...","key_hex":"...","method":"..."}
        self.patterns = {}       # encrypted_hex_prefix → {"seen_in":[...],"decrypts_to":"..."}
        self._load()

    def _load(self):
        if self.INTEL_FILE.exists():
            try:
                data = json.loads(self.INTEL_FILE.read_text())
                self.credentials = data.get("credentials", [])
                self.cribs = data.get("cribs", [])
                self.patterns = data.get("patterns", {})
                log.info(f"Intel loaded: {len(self.credentials)} creds, {len(self.cribs)} cribs, {len(self.patterns)} patterns")
            except Exception as e:
                log.warning(f"Intel load failed: {e}")

    def _save(self):
        try:
            self.INTEL_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.INTEL_FILE.write_text(json.dumps({
                "credentials": self.credentials[-500:],  # Cap at 500
                "cribs": self.cribs[-200:],               # Cap at 200
                "patterns": dict(list(self.patterns.items())[-300:]),
                "last_updated": datetime.now().isoformat(),
            }, indent=2))
        except Exception as e:
            log.error(f"Intel save failed: {e}")

    def harvest_from_proofs(self, stream_id: str, data: bytes, proofs: list, capture: str = ""):
        """Extract credentials, cribs, and patterns from successful proofs."""
        added = 0
        for p in proofs:
            if not p.get("success"): continue
            method = p.get("method", "")

            # ── Harvest credentials from sensitive data findings ──
            for lbl, val in p.get("sensitive", []):
                cred = {"type": lbl, "value": val, "source": stream_id,
                        "capture": capture, "method": method,
                        "time": datetime.now().isoformat()}
                if not any(c["type"] == lbl and c["value"] == val for c in self.credentials):
                    self.credentials.append(cred)
                    added += 1
                    log.debug(f"Intel: credential {lbl}={val[:20]}... from {stream_id}")

            # ── Harvest PAN numbers from previews ──
            preview = p.get("preview", "") or p.get("final_preview", "")
            for pan_match in re.findall(r'(\d{6})\*{4,6}(\d{4})', preview):
                pan_masked = f"{pan_match[0]}******{pan_match[1]}"
                cred = {"type": "PAN", "value": pan_masked, "source": stream_id,
                        "capture": capture, "time": datetime.now().isoformat()}
                if not any(c["type"] == "PAN" and c["value"] == pan_masked for c in self.credentials):
                    self.credentials.append(cred); added += 1

            # ── Harvest String Extraction samples as plaintext cribs ──
            if method == "String Extraction" and "samples" in p:
                for sample in p["samples"]:
                    sample = sample.strip()
                    # Require longer strings with real word-like content
                    if len(sample) < 12 or len(sample) > 80: continue
                    if not re.match(r'^[\x20-\x7e]+$', sample): continue
                    # Must contain actual words (not random printable chars)
                    alpha = sum(1 for c in sample if c.isalpha())
                    if alpha < len(sample) * 0.4: continue
                    # Filter protocol noise
                    if any(x in sample.lower() for x in ['sha256','sha384','curve25519',
                        'openssh','libssh','poly1305','chacha20']): continue
                    crib = {"plaintext": sample, "plaintext_hex": sample.encode().hex(),
                            "ciphertext_hex": "", "method": "String Extraction (raw)",
                            "source": stream_id, "capture": capture}
                    if not any(c["plaintext"] == sample for c in self.cribs):
                        self.cribs.append(crib); added += 1

            # ── Build plaintext↔ciphertext cribs from HIGH-QUALITY decryption proofs only ──
            if method in ("XOR Key Recovery", "Caesar/Shift", "Known-Plaintext Attack") and "key_hex" in p:
                # Only store cribs from decryptions that actually produced readable output
                if p.get("printable", 0) < 0.7: continue  # Skip garbled partial decryptions
                key_hex = p["key_hex"]
                for pattern in re.findall(r'[a-zA-Z][a-zA-Z0-9_=&/]{7,39}', preview):
                    # Must start with a letter and be mostly alphanumeric
                    alpha = sum(1 for c in pattern if c.isalpha())
                    if alpha < len(pattern) * 0.5: continue
                    pt_bytes = pattern.encode('ascii', errors='ignore')
                    if method in ("XOR Key Recovery", "Known-Plaintext Attack"):
                        try:
                            key = bytes.fromhex(key_hex)
                            ct_bytes = bytes(pt_bytes[i] ^ key[i % len(key)] for i in range(len(pt_bytes)))
                            crib = {"plaintext": pattern, "plaintext_hex": pt_bytes.hex(),
                                    "ciphertext_hex": ct_bytes.hex(),
                                    "key_hex": key_hex, "method": method,
                                    "source": stream_id, "capture": capture}
                            if not any(c["plaintext"] == pattern and c["key_hex"] == key_hex for c in self.cribs):
                                self.cribs.append(crib); added += 1
                        except: pass
                    elif method == "Caesar/Shift" and "shift" in p:
                        shift = p["shift"]
                        ct_bytes = bytes((b + shift) % 256 for b in pt_bytes)
                        crib = {"plaintext": pattern, "plaintext_hex": pt_bytes.hex(),
                                "ciphertext_hex": ct_bytes.hex(),
                                "shift": shift, "method": method,
                                "source": stream_id, "capture": capture}
                        if not any(c["plaintext"] == pattern and c.get("shift") == shift for c in self.cribs):
                            self.cribs.append(crib); added += 1

            # ── Store encrypted pattern fingerprints ──
            # If we decrypted something, store first 16 bytes of ciphertext → plaintext mapping
            if method in ("XOR Key Recovery", "Caesar/Shift", "Known-Plaintext Attack") and p.get("printable", 0) > 0.5:
                ct_prefix = data[:16].hex()
                if ct_prefix not in self.patterns:
                    self.patterns[ct_prefix] = {
                        "decrypts_to": preview[:50],
                        "method": method,
                        "key_hex": p.get("key_hex", ""),
                        "seen_in": [{"stream": stream_id, "capture": capture,
                                     "time": datetime.now().isoformat()}]
                    }
                else:
                    # Same encrypted prefix seen again!
                    existing = self.patterns[ct_prefix]
                    if not any(s["stream"] == stream_id for s in existing["seen_in"]):
                        existing["seen_in"].append({"stream": stream_id, "capture": capture,
                                                     "time": datetime.now().isoformat()})
                        added += 1

        # ── Scan raw data directly for plaintext credentials ──
        raw_text = data[:4096].decode('ascii', errors='ignore')
        ssh_noise_domains={'libssh.org','openssh.com','ietf.org'}
        for pat, lbl in [
            (r'(?i)(?:user(?:name)?|login|email)\s*[=:]\s*([^\s&;,]{3,})', "USERNAME"),
            (r'(?i)(?:pass(?:word|wd)?|secret|pin)\s*[=:]\s*([^\s&;,]{3,})', "PASSWORD"),
            (r'(?i)(?:bearer|token|api[_-]?key|session[_-]?id)\s*[=:]\s*([^\s&;,]{8,})', "AUTH_TOKEN"),
            (r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', "EMAIL"),
        ]:
            for m in re.findall(pat, raw_text)[:3]:
                val = m if isinstance(m, str) else m[-1]
                val = re.sub(r'(\d{6})\d{6,10}(\d{4})', r'\1******\2', val)
                # Filter SSH/protocol noise
                if lbl=="EMAIL" and any(d in val.lower() for d in ssh_noise_domains): continue
                # Filter timestamps as usernames
                if re.match(r'^20\d{10,}$', val): continue
                if len(val) < 3: continue
                cred = {"type": lbl, "value": val[:64], "source": stream_id,
                        "capture": capture, "method": "raw_scan",
                        "time": datetime.now().isoformat()}
                if not any(c["type"] == lbl and c["value"] == val[:64] for c in self.credentials):
                    self.credentials.append(cred); added += 1
                # Only store substantial values as cribs (min 10 chars)
                if len(val) >= 10:
                    crib = {"plaintext": val[:64], "plaintext_hex": val[:64].encode().hex(),
                            "ciphertext_hex": "", "method": "raw_scan",
                            "source": stream_id, "capture": capture}
                    if not any(c["plaintext"] == val[:64] for c in self.cribs):
                        self.cribs.append(crib); added += 1

        if added > 0: self._save()

    def get_auto_cribs(self) -> list:
        """Return high-quality plaintext strings for auto known-plaintext attacks.
        Filters out garbled XOR fragments and protocol noise."""
        cribs = set()
        # From stored credentials — only real values
        for c in self.credentials:
            val = c.get("value", "")
            for part in re.split(r'[=:]\s*', val):
                part = part.strip()
                if len(part) < 6 or len(part) > 64: continue
                # Must look like a real word/value (majority letters/digits)
                alnum = sum(1 for ch in part if ch.isalnum())
                if alnum < len(part) * 0.6: continue
                # Filter timestamps
                if re.match(r'^20\d{10,}$', part): continue
                # Filter SSH/protocol noise
                if any(x in part.lower() for x in ['openssh','libssh','sha256','sha384',
                    'curve25519','poly1305','chacha20','aes128','aes256']): continue
                cribs.add(part)
        # From stored cribs — only from high-quality decryptions
        for c in self.cribs:
            pt = c.get("plaintext", "")
            method = c.get("method", "")
            if len(pt) < 8 or len(pt) > 64: continue
            # Cribs from XOR recovery are often garbled — require they look like real text
            if "XOR" in method or "Known-Plaintext" in method:
                # Must be mostly lowercase letters or known structure (key=value, paths, etc.)
                alpha = sum(1 for ch in pt if ch.isalpha())
                if alpha < len(pt) * 0.7: continue
                # Reject random-looking strings (high character diversity relative to length)
                if len(pt) < 15 and len(set(pt)) > len(pt) * 0.9: continue
            elif method in ("String Extraction (raw)", "raw_scan"):
                # String extraction cribs are generally good — just filter short ones
                if len(pt) < 10: continue
            # Final check: must have some repeating character patterns (real text does)
            if len(pt) >= 12 and len(set(pt.lower())) > len(pt) * 0.95: continue
            cribs.add(pt)
        return [c.encode('utf-8', errors='ignore') for c in cribs if len(c) >= 8]

    def check_pattern(self, data: bytes) -> list:
        """Check if this data matches any known encrypted patterns."""
        if len(data) < 16: return []
        findings = []
        prefix = data[:16].hex()
        if prefix in self.patterns:
            pat = self.patterns[prefix]
            findings.append({
                "method": "Known Pattern Match",
                "success": True,
                "confidence": 0.8,
                "printable": 0.0,
                "pattern_prefix": prefix,
                "previously_decrypts_to": pat["decrypts_to"],
                "original_method": pat["method"],
                "key_hex": pat.get("key_hex", ""),
                "times_seen": len(pat["seen_in"]),
                "first_seen": pat["seen_in"][0] if pat["seen_in"] else {},
            })
            log.info(f"Pattern match: prefix {prefix[:16]} seen {len(pat['seen_in'])} times")
        return findings

    def get_report(self) -> dict:
        """Summary of collected intelligence."""
        cred_types = collections.Counter(c["type"] for c in self.credentials)
        capture_coverage = set()
        for c in self.credentials: capture_coverage.add(c.get("capture", ""))
        for c in self.cribs: capture_coverage.add(c.get("capture", ""))
        capture_coverage.discard("")

        # Find credentials seen in multiple captures
        cred_cross = {}
        for c in self.credentials:
            key = f"{c['type']}:{c['value']}"
            cred_cross.setdefault(key, set()).add(c.get("capture", ""))
        multi_capture_creds = {k: v for k, v in cred_cross.items() if len(v) > 1}

        # Patterns seen multiple times
        recurring = {k: v for k, v in self.patterns.items() if len(v.get("seen_in", [])) > 1}

        return {
            "total_credentials": len(self.credentials),
            "credential_types": dict(cred_types),
            "total_cribs": len(self.cribs),
            "total_patterns": len(self.patterns),
            "captures_analyzed": len(capture_coverage),
            "recurring_patterns": len(recurring),
            "cross_capture_credentials": {k: list(v) for k, v in multi_capture_creds.items()},
        }

    def clear(self):
        self.credentials = []; self.cribs = []; self.patterns = {}
        if self.INTEL_FILE.exists(): self.INTEL_FILE.unlink()
        log.info("Intel vault cleared")

# ═══════════════════════════════════════════════════════════════
#  Input Parser — handles multiple formats
# ═══════════════════════════════════════════════════════════════
class InputParser:
    """Parses any file or text input into raw bytes for analysis."""

    @staticmethod
    def from_file(path: str) -> tuple:
        p=Path(path); ext=p.suffix.lower()
        if ext in ('.pcap','.pcapng','.cap'): return None,"pcap"
        data=p.read_bytes()
        # JSON (NetSentinel exports, generic)
        if ext=='.json':
            try:
                j=json.loads(data)
                payloads=[]
                items=j if isinstance(j,list) else [j]
                for item in items:
                    if isinstance(item,dict):
                        for key in ['payload','data','raw','hex','bytes','encrypted','ciphertext','body','content']:
                            if key in item:
                                parsed=InputParser.parse_text(str(item[key]))
                                if parsed: payloads.append(parsed)
                if payloads: return b''.join(payloads),f"JSON ({len(payloads)} payloads)"
            except: pass
            return data,f"JSON raw ({len(data)}B)"
        # CSV
        if ext=='.csv':
            try:
                text=data.decode('utf-8',errors='ignore')
                payloads=[]
                for row in csv.reader(io.StringIO(text)):
                    for cell in row:
                        parsed=InputParser.parse_text(cell.strip())
                        if parsed and len(parsed)>=8: payloads.append(parsed)
                if payloads: return b''.join(payloads),f"CSV ({len(payloads)} payloads)"
            except: pass
            return data,f"CSV raw ({len(data)}B)"
        # Text / hex / encrypted text
        if ext in ('.txt','.hex','.enc','.encrypted','.cipher','.key','.log','.dat'):
            text=data.decode('utf-8',errors='ignore')
            # Try hex first
            parsed=InputParser._try_hex(text)
            if parsed: return parsed,f"Hex decoded ({len(parsed)}B from {ext})"
            # Try base64
            parsed=InputParser._try_b64(text)
            if parsed: return parsed,f"Base64 decoded ({len(parsed)}B from {ext})"
            # It's just text — analyze it as raw bytes (maybe it IS the ciphertext)
            return data,f"Text file ({len(data)}B) — analyzing as raw bytes"
        # Binary
        return data,f"Binary ({len(data)}B)"

    @staticmethod
    def parse_text(text: str) -> Optional[bytes]:
        """Parse any text input into bytes. Tries hex, base64, URL-encoded, then raw."""
        text=text.strip()
        if not text: return None
        # Try hex
        r=InputParser._try_hex(text)
        if r: return r
        # Try base64
        r=InputParser._try_b64(text)
        if r: return r
        # Try URL decode
        if '%' in text:
            try:
                from urllib.parse import unquote_to_bytes
                dec=unquote_to_bytes(text)
                if dec!=text.encode(): return dec
            except: pass
        # Raw text — encode as bytes
        return text.encode('utf-8')

    @staticmethod
    def _try_hex(text):
        # Remove common hex formatting: 0x, spaces, colons, dashes, newlines
        clean=re.sub(r'[\s:.\-\n\r]','',text)
        clean=re.sub(r'0x','',clean,flags=re.IGNORECASE)
        if len(clean)>=8 and len(clean)%2==0 and all(c in '0123456789abcdefABCDEF' for c in clean):
            try: return bytes.fromhex(clean)
            except: pass
        return None

    @staticmethod
    def _try_b64(text):
        clean=text.strip()
        if re.match(r'^[A-Za-z0-9+/=\s]{16,}$',clean):
            try:
                dec=base64.b64decode(re.sub(r'\s','',clean),validate=True)
                if len(dec)<4: return None
                # Validate: decoded should look like real data, not worse than input
                # If input is readable text, decoding to garbage means it's not actually base64
                input_printable=sum(1 for c in clean if 32<=ord(c)<=126)/max(len(clean),1)
                dec_printable=sum(1 for b in dec if 32<=b<=126 or b in(9,10,13))/max(len(dec),1)
                # If input is highly printable text but decoded is mostly binary garbage, reject
                if input_printable>0.9 and dec_printable<0.5 and len(clean)<200:
                    return None  # Probably just text that happens to look like base64
                return dec
            except: pass
        return None

# ═══════════════════════════════════════════════════════════════
#  Debug / Export
# ═══════════════════════════════════════════════════════════════
class DebugExporter:
    """Creates debug snapshots and session reports for troubleshooting."""

    @staticmethod
    def export_debug_snapshot():
        """Creates a debug ZIP with logs, settings, model info, and stats."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_dir = DEBUG_DIR / f"snapshot_{ts}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        # Copy recent logs
        for lf in LOGS_DIR.glob("*.log*"):
            shutil.copy2(lf, snap_dir)

        # System info
        info = {
            "timestamp": ts,
            "python": sys.version,
            "platform": sys.platform,
            "scapy": HAS_SCAPY,
            "app_dir": str(APP_DIR),
            "model_exists": (MODELS_DIR / "models.joblib").exists(),
            "feedback_samples": sum(1 for _ in SAMPLES_DIR.rglob("*.bin")) if SAMPLES_DIR.exists() else 0,
        }
        (snap_dir / "system_info.json").write_text(json.dumps(info, indent=2))

        # Method stats
        ms = FEEDBACK_DIR / "method_stats.json"
        if ms.exists(): shutil.copy2(ms, snap_dir)

        # Settings
        sp = APP_DIR / "settings.json"
        if sp.exists(): shutil.copy2(sp, snap_dir)

        # Zip it
        zip_path = DEBUG_DIR / f"debug_{ts}"
        shutil.make_archive(str(zip_path), 'zip', snap_dir)
        shutil.rmtree(snap_dir)

        log.info(f"Debug snapshot: {zip_path}.zip")
        return f"{zip_path}.zip"

    @staticmethod
    def export_session(results: list, filename: str = None):
        """Export analysis results as JSON for review."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = filename or f"session_{ts}.json"
        path = EXPORTS_DIR / fn

        export = {
            "timestamp": datetime.now().isoformat(),
            "app_version": "1.0",
            "results_count": len(results),
            "results": results,
        }

        # Clean for JSON serialization
        def clean(obj):
            if isinstance(obj, bytes): return obj.hex()[:200]
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return str(obj)

        path.write_text(json.dumps(export, indent=2, default=clean))
        log.info(f"Session exported: {path}")
        return str(path)


# ═══════════════════════════════════════════════════════════════
#  GUI Application
# ═══════════════════════════════════════════════════════════════
class CryptoAuditGUI:
    BG="#1a1a2e"; BG2="#16213e"; BG3="#0f3460"; FG="#e6e6e6"; FG2="#a0a0b0"
    ACCENT="#00b4d8"; RED="#ff4444"; YELLOW="#ffaa00"; GREEN="#00cc66"
    CARD_BG="#1e2746"; ENTRY_BG="#0d1b2a"; BTN_BG="#0f3460"

    def __init__(self):
        self.root=tk.Tk()
        self.root.title("CryptoAudit ML — Encryption Analyzer")
        self.root.geometry("1320x860")
        self.root.configure(bg=self.BG)
        self.root.minsize(1000,700)

        self.models=CryptoMLModels()
        self.settings=self._load_settings()
        self.validator=AIValidator(self.settings["ollama_url"],self.settings["ollama_model"])
        self.feedback=FeedbackCollector()
        self.harvester=KeyHarvester()
        self.intel=IntelCollector()
        self._lock=threading.Lock()
        self._live_on=False
        self._session_results=[]

        self.model_loaded=self.models.load()
        self._setup_styles()
        self._build_ui()
        self._update_status()
        if not self.model_loaded:
            self._auto_train()

    def _load_settings(self):
        defaults={"ollama_url":"http://localhost:11434","ollama_model":"qwen2.5:14b",
            "samples":500,"auto_decrypt":True,"auto_validate":True,
            "capture_iface":"","capture_bpf":DEFAULT_BPF}
        sp=APP_DIR/"settings.json"
        if sp.exists():
            try: defaults.update(json.loads(sp.read_text()))
            except: pass
        return defaults

    def _save_settings(self):
        (APP_DIR/"settings.json").write_text(json.dumps(self.settings,indent=2))

    def _auto_train(self):
        self._set_status("First launch — training models...")
        threading.Thread(target=self._auto_train_bg,daemon=True).start()

    def _auto_train_bg(self):
        with self._lock:
            p,l=TrainingDataGenerator.generate(300)
            self.models.train(p,l); self.models.save(); self.model_loaded=True
        self.root.after(0,lambda:self._set_status("Models ready"))
        self.root.after(0,self._refresh_dashboard)

    def _ensure_trained(self):
        if self.models.is_trained: return
        with self._lock:
            if self.models.is_trained: return
            p,l=TrainingDataGenerator.generate(300)
            self.models.train(p,l); self.models.save(); self.model_loaded=True

    def _setup_styles(self):
        s=ttk.Style(); s.theme_use("clam")
        s.configure(".",background=self.BG,foreground=self.FG,fieldbackground=self.ENTRY_BG,borderwidth=0)
        s.configure("TNotebook",background=self.BG,borderwidth=0)
        s.configure("TNotebook.Tab",background=self.BG2,foreground=self.FG2,padding=[16,8],font=("Segoe UI",10))
        s.map("TNotebook.Tab",background=[("selected",self.BG3)],foreground=[("selected",self.ACCENT)])
        s.configure("TFrame",background=self.BG)
        s.configure("TLabel",background=self.BG,foreground=self.FG,font=("Segoe UI",10))
        s.configure("TButton",background=self.BTN_BG,foreground=self.FG,font=("Segoe UI",10),padding=[12,6])
        s.map("TButton",background=[("active","#1a5276")])
        s.configure("Accent.TButton",background=self.ACCENT,foreground="#000",font=("Segoe UI",10,"bold"))
        s.map("Accent.TButton",background=[("active","#0096b7")])
        s.configure("TEntry",fieldbackground=self.ENTRY_BG,foreground=self.FG,insertcolor=self.FG)
        s.configure("TCheckbutton",background=self.BG,foreground=self.FG)
        s.configure("Treeview",background=self.ENTRY_BG,foreground=self.FG,fieldbackground=self.ENTRY_BG,
            font=("Consolas",9),rowheight=24)
        s.configure("Treeview.Heading",background=self.BG3,foreground=self.ACCENT,font=("Segoe UI",9,"bold"))
        s.map("Treeview",background=[("selected",self.BG3)])
        s.configure("TProgressbar",background=self.ACCENT,troughcolor=self.ENTRY_BG)

    def _build_ui(self):
        hdr=tk.Frame(self.root,bg=self.BG2,height=50); hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr,text="🔐 CryptoAudit ML",font=("Segoe UI",16,"bold"),fg=self.ACCENT,bg=self.BG2).pack(side="left",padx=16)
        tk.Label(hdr,text="AI-Powered Encryption Analyzer",font=("Segoe UI",10),fg=self.FG2,bg=self.BG2).pack(side="left",padx=8)
        ttk.Button(hdr,text="📦 Export Debug",command=self._export_debug).pack(side="right",padx=8,pady=8)
        ttk.Button(hdr,text="💾 Export Session",command=self._export_session).pack(side="right",padx=4,pady=8)

        self.nb=ttk.Notebook(self.root)
        self.nb.pack(fill="both",expand=True,padx=8,pady=(8,0))

        self._tab_dash(); self._tab_pcap(); self._tab_live(); self._tab_payload(); self._tab_train(); self._tab_settings()

        sf=tk.Frame(self.root,bg=self.BG2,height=28); sf.pack(fill="x"); sf.pack_propagate(False)
        self.status_lbl=tk.Label(sf,text="",font=("Consolas",9),fg=self.FG2,bg=self.BG2,anchor="w")
        self.status_lbl.pack(side="left",padx=12,fill="x",expand=True)
        self.ollama_lbl=tk.Label(sf,text="",font=("Consolas",9),bg=self.BG2,anchor="e")
        self.ollama_lbl.pack(side="right",padx=12)

    def _set_status(self,msg): self.status_lbl.config(text=msg)

    def _update_status(self):
        m="Models: ✓" if self.model_loaded else "Models: training..."
        if self.models.history:
            m+=f" (RF={self.models.history[-1].get('rf',0):.1%})"
        self._set_status(m)
        def chk():
            a=AIValidator(self.settings["ollama_url"],self.settings["ollama_model"])._check()
            self.root.after(0,lambda:self.ollama_lbl.config(
                text=f"Ollama: {'✓' if a else '✗'}",fg=self.GREEN if a else self.FG2))
        threading.Thread(target=chk,daemon=True).start()

    def _card(self,parent,title,val,col):
        f=tk.Frame(parent,bg=self.CARD_BG,bd=1,relief="solid",padx=16,pady=12)
        f.grid(row=0,column=col,padx=6,sticky="nsew")
        tk.Label(f,text=title,font=("Segoe UI",9),fg=self.FG2,bg=self.CARD_BG).pack(anchor="w")
        l=tk.Label(f,text=val,font=("Segoe UI",18,"bold"),fg=self.FG,bg=self.CARD_BG)
        l.pack(anchor="w",pady=(4,0)); return l

    # ── Dashboard ──
    def _tab_dash(self):
        t=ttk.Frame(self.nb); self.nb.add(t,text="  Dashboard  ")
        cards=tk.Frame(t,bg=self.BG); cards.pack(fill="x",padx=16,pady=12)
        self.c_model=self._card(cards,"Model","✓" if self.model_loaded else "...",0)
        self.c_acc=self._card(cards,"Accuracy",
            f"{self.models.history[-1]['rf']:.1%}" if self.models.history else "—",1)
        fb_n=sum(1 for _ in SAMPLES_DIR.rglob("*.bin")) if SAMPLES_DIR.exists() else 0
        self.c_fb=self._card(cards,"Feedback",str(fb_n),2)
        self.c_sess=self._card(cards,"Session","0 results",3)
        self.c_keys=self._card(cards,"Key Vault","0 keys",4)
        for i in range(5): cards.columnconfigure(i,weight=1)

        # Vault viewer buttons
        vbtn=tk.Frame(t,bg=self.BG); vbtn.pack(fill="x",padx=16,pady=(0,8))
        ttk.Button(vbtn,text="🔑 View Key Vault",command=lambda:self._show_vault("keys")).pack(side="left",padx=4)
        ttk.Button(vbtn,text="🧠 View Intel Vault",command=lambda:self._show_vault("intel")).pack(side="left",padx=4)
        ttk.Button(vbtn,text="📊 Full Report",command=lambda:self._show_vault("report")).pack(side="left",padx=4)

        # Method reliability
        mf=tk.LabelFrame(t,text=" Method Reliability ",bg=self.CARD_BG,fg=self.ACCENT,
            font=("Segoe UI",11,"bold"),bd=1,relief="solid")
        mf.pack(fill="both",expand=True,padx=16,pady=(0,12))
        self.meth_tree=ttk.Treeview(mf,columns=("att","conf","likely","fp","score"),show="headings",height=8)
        for c,w in [("att",70),("conf",80),("likely",70),("fp",60),("score",90)]:
            self.meth_tree.heading(c,text=c.title()); self.meth_tree.column(c,width=w,anchor="center")
        self.meth_tree.pack(fill="both",expand=True,padx=8,pady=8)
        self._refresh_dashboard()

    def _refresh_dashboard(self):
        for i in self.meth_tree.get_children(): self.meth_tree.delete(i)
        for m,s in self.feedback.stats.items():
            att=s.get("attempts",0)
            sc=(s.get("confirmed",0)+0.5*s.get("likely",0))/max(att,1)
            self.meth_tree.insert("","end",text=m,values=(att,s.get("confirmed",0),
                s.get("likely",0),s.get("fp",0),f"{sc:.0%}"))
        if self.model_loaded:
            self.c_model.config(text="✓ Loaded")
        if self.models.history:
            self.c_acc.config(text=f"{self.models.history[-1].get('rf',0):.1%}")
        self.c_keys.config(text=f"{len(self.harvester.keys)} keys")
        reuse=self.harvester.get_reuse_report()
        if reuse["keys_reused_across_streams"]>0:
            self.c_keys.config(text=f"{len(self.harvester.keys)} keys ({reuse['keys_reused_across_streams']} reused)")
        # Intel stats
        intel_r=self.intel.get_report()
        intel_total=intel_r["total_credentials"]+intel_r["total_cribs"]+intel_r["total_patterns"]
        if intel_total>0:
            self.c_fb.config(text=f"{sum(1 for _ in SAMPLES_DIR.rglob('*.bin')) if SAMPLES_DIR.exists() else 0} fb / {intel_total} intel")

    def _show_vault(self, mode):
        """Show vault contents in a popup window."""
        win=tk.Toplevel(self.root)
        win.title({"keys":"Key Vault","intel":"Intel Vault","report":"Full Intelligence Report"}.get(mode,"Vault"))
        win.geometry("850x600"); win.configure(bg=self.BG)

        # Header
        hdr=tk.Frame(win,bg=self.BG2,height=40); hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr,text={"keys":"🔑 Key Vault","intel":"🧠 Intel Vault","report":"📊 Full Report"}.get(mode,""),
            font=("Segoe UI",14,"bold"),fg=self.ACCENT,bg=self.BG2).pack(side="left",padx=12)
        ttk.Button(hdr,text="Export to File",command=lambda:self._export_vault(mode)).pack(side="right",padx=8,pady=4)

        # Content
        txt=scrolledtext.ScrolledText(win,wrap="word",bg=self.ENTRY_BG,fg=self.FG,
            font=("Consolas",9),insertbackground=self.FG,bd=0,padx=12,pady=8)
        txt.pack(fill="both",expand=True,padx=8,pady=8)

        if mode=="keys":
            txt.insert("end",f"{'═'*55}\n  KEY VAULT — {len(self.harvester.keys)} keys\n{'═'*55}\n\n")
            if not self.harvester.keys:
                txt.insert("end","  No keys collected yet.\n  Analyze a PCAP or run a live capture to start harvesting.\n")
            else:
                # Group by source capture
                by_capture=collections.defaultdict(list)
                for k in self.harvester.keys:
                    by_capture[k.get("capture","unknown")].append(k)
                for cap,keys in by_capture.items():
                    txt.insert("end",f"  ── Capture: {cap} ({len(keys)} keys) ──\n")
                    for k in keys:
                        txt.insert("end",f"\n    Key:      {k['key_hex'][:40]}{'...' if len(k['key_hex'])>40 else ''}\n")
                        txt.insert("end",f"    Length:   {k['key_len']} bytes\n")
                        txt.insert("end",f"    Method:   {k['method']}\n")
                        txt.insert("end",f"    Source:   {k['source']}\n")
                        txt.insert("end",f"    Conf:     {k['confidence']:.0%}  |  Readable: {k['printable']:.0%}\n")
                        txt.insert("end",f"    Harvest:  {k.get('harvested_at','?')}\n")
                    txt.insert("end","\n")

                # Reuse report
                rr=self.harvester.get_reuse_report()
                if rr["keys_reused_across_streams"]>0:
                    txt.insert("end",f"  {'═'*55}\n  ⚠ KEY REUSE DETECTED\n  {'═'*55}\n")
                    for kh,info in rr["reuse_details"].items():
                        txt.insert("end",f"\n    Key {kh[:24]}...\n")
                        for s in info["streams"]:
                            txt.insert("end",f"      • {s}\n")
                if rr.get("cross_capture_reuse"):
                    txt.insert("end",f"\n  🔴 CROSS-CAPTURE REUSE\n")
                    for kh,info in rr["cross_capture_reuse"].items():
                        txt.insert("end",f"    Key {kh[:24]}... → {', '.join(info['captures'])}\n")

        elif mode=="intel":
            ir=self.intel.get_report()
            txt.insert("end",f"{'═'*55}\n  INTEL VAULT\n{'═'*55}\n\n")

            # Credentials
            txt.insert("end",f"  ── Credentials ({ir['total_credentials']}) ──\n")
            if not self.intel.credentials:
                txt.insert("end","  None collected yet.\n\n")
            else:
                by_type=collections.defaultdict(list)
                for c in self.intel.credentials:
                    by_type[c["type"]].append(c)
                for typ,creds in by_type.items():
                    txt.insert("end",f"\n    {typ} ({len(creds)}):\n")
                    for c in creds[:20]:
                        txt.insert("end",f"      {c['value'][:50]}")
                        if c.get("capture"): txt.insert("end",f"  [{c['capture']}]")
                        txt.insert("end","\n")
                    if len(creds)>20: txt.insert("end",f"      ... and {len(creds)-20} more\n")
                txt.insert("end","\n")

            # Cribs
            txt.insert("end",f"  ── Plaintext Cribs ({ir['total_cribs']}) ──\n")
            if not self.intel.cribs:
                txt.insert("end","  None collected yet.\n\n")
            else:
                for c in self.intel.cribs[:30]:
                    txt.insert("end",f"    \"{c['plaintext'][:45]}\"")
                    if c.get("capture"): txt.insert("end",f"  [{c['capture']}]")
                    txt.insert("end","\n")
                if len(self.intel.cribs)>30:
                    txt.insert("end",f"    ... and {len(self.intel.cribs)-30} more\n")
                txt.insert("end","\n")

            # Patterns
            txt.insert("end",f"  ── Encrypted Patterns ({ir['total_patterns']}) ──\n")
            for prefix,pat in list(self.intel.patterns.items())[:20]:
                times=len(pat.get("seen_in",[]))
                txt.insert("end",f"    {prefix[:24]}... → \"{pat.get('decrypts_to','')[:35]}\"")
                if times>1: txt.insert("end",f"  (seen {times}×)")
                txt.insert("end","\n")

            # Cross-capture
            if ir.get("cross_capture_credentials"):
                txt.insert("end",f"\n  🔴 CREDENTIALS SEEN IN MULTIPLE CAPTURES\n")
                for cred,caps in ir["cross_capture_credentials"].items():
                    txt.insert("end",f"    {cred[:50]} → {', '.join(caps)}\n")

        elif mode=="report":
            kr=self.harvester.get_reuse_report()
            ir=self.intel.get_report()
            txt.insert("end",f"{'═'*55}\n  FULL INTELLIGENCE REPORT\n  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'═'*55}\n\n")

            txt.insert("end",f"  Key Vault:       {kr['total_keys_harvested']} keys ({kr['unique_keys']} unique)\n")
            txt.insert("end",f"  Key Reuse:       {kr['keys_reused_across_streams']} keys shared across streams\n")
            txt.insert("end",f"  Cross-Capture:   {len(kr.get('cross_capture_reuse',{}))} keys shared across captures\n\n")

            txt.insert("end",f"  Credentials:     {ir['total_credentials']}\n")
            if ir["credential_types"]:
                txt.insert("end",f"    Types:         {', '.join(f'{v}×{k}' for k,v in ir['credential_types'].items())}\n")
            txt.insert("end",f"  Auto-Cribs:      {ir['total_cribs']}\n")
            txt.insert("end",f"  Patterns:        {ir['total_patterns']} ({ir['recurring_patterns']} recurring)\n")
            txt.insert("end",f"  Captures:        {ir['captures_analyzed']} analyzed\n\n")

            if kr["keys_reused_across_streams"]>0:
                txt.insert("end",f"  {'─'*50}\n  KEY REUSE DETAILS\n  {'─'*50}\n")
                for kh,info in kr["reuse_details"].items():
                    txt.insert("end",f"  Key {kh[:24]}... → {info['count']} streams\n")

            if ir.get("cross_capture_credentials"):
                txt.insert("end",f"\n  {'─'*50}\n  CROSS-CAPTURE CREDENTIALS\n  {'─'*50}\n")
                for cred,caps in ir["cross_capture_credentials"].items():
                    txt.insert("end",f"  {cred[:50]} → {', '.join(caps)}\n")

            txt.insert("end",f"\n  {'─'*50}\n  ALL CREDENTIALS\n  {'─'*50}\n")
            for c in self.intel.credentials[:50]:
                txt.insert("end",f"  {c['type']:15s} {c['value'][:40]:40s} [{c.get('capture','')}]\n")

    def _export_vault(self, mode):
        """Export vault contents to a JSON file."""
        ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        if mode=="keys":
            data={"keys":[{k:v for k,v in e.items() if k!="key"} for e in self.harvester.keys],
                  "reuse":self.harvester.get_reuse_report()}
            fn=f"key_vault_export_{ts}.json"
        elif mode=="intel":
            data={"credentials":self.intel.credentials,"cribs":self.intel.cribs,
                  "patterns":self.intel.patterns,"report":self.intel.get_report()}
            fn=f"intel_export_{ts}.json"
        else:
            data={"keys":self.harvester.get_reuse_report(),
                  "intel":self.intel.get_report(),
                  "credentials":self.intel.credentials[:100],
                  "cribs":self.intel.cribs[:100]}
            fn=f"full_report_{ts}.json"
        path=EXPORTS_DIR/fn
        path.write_text(json.dumps(data,indent=2,default=str))
        messagebox.showinfo("Exported",f"Saved to:\n{path}")

    # ── PCAP ──
    def _tab_pcap(self):
        t=ttk.Frame(self.nb); self.nb.add(t,text="  PCAP Analysis  ")
        ctrl=tk.Frame(t,bg=self.BG); ctrl.pack(fill="x",padx=16,pady=12)
        tk.Label(ctrl,text="File:",fg=self.FG,bg=self.BG).pack(side="left")
        self.pcap_path=tk.StringVar()
        ttk.Entry(ctrl,textvariable=self.pcap_path,width=45).pack(side="left",padx=4)
        ttk.Button(ctrl,text="Browse",command=lambda:self.pcap_path.set(
            filedialog.askopenfilename(filetypes=[("PCAP","*.pcap *.pcapng *.cap"),("All","*.*")]) or self.pcap_path.get()
        )).pack(side="left",padx=4)
        tk.Label(ctrl,text="Ports:",fg=self.FG,bg=self.BG).pack(side="left",padx=(12,4))
        self.pcap_ports=tk.StringVar(value="all")
        ttk.Entry(ctrl,textvariable=self.pcap_ports,width=15).pack(side="left")
        tk.Label(ctrl,text="(\"all\" or 443,80,...)",fg=self.FG2,bg=self.BG,font=("Segoe UI",8)).pack(side="left",padx=4)
        ttk.Button(ctrl,text="▶ Analyze",style="Accent.TButton",command=self._run_pcap).pack(side="right")

        # Progress bar for PCAP loading
        self.pcap_prog_frame=tk.Frame(t,bg=self.BG)
        self.pcap_prog_frame.pack(fill="x",padx=16,pady=(0,4))
        self.pcap_prog=ttk.Progressbar(self.pcap_prog_frame,mode="determinate",maximum=100)
        self.pcap_prog.pack(side="left",fill="x",expand=True)
        self.pcap_prog_label=tk.Label(self.pcap_prog_frame,text="",fg=self.FG2,bg=self.BG,
            font=("Consolas",8),width=40,anchor="w")
        self.pcap_prog_label.pack(side="right",padx=(8,0))

        pn=tk.PanedWindow(t,orient="horizontal",bg=self.BG,sashwidth=4)
        pn.pack(fill="both",expand=True,padx=16,pady=(0,12))
        lf=tk.Frame(pn,bg=self.CARD_BG); pn.add(lf,width=520)
        self.pcap_tree=ttk.Treeview(lf,columns=("v","c","b","ae"),show="tree headings")
        self.pcap_tree.heading("#0",text="Stream"); self.pcap_tree.heading("v",text="Verdict")
        self.pcap_tree.heading("c",text="Conf"); self.pcap_tree.heading("b",text="Bytes"); self.pcap_tree.heading("ae",text="AE")
        self.pcap_tree.column("#0",width=200); self.pcap_tree.column("v",width=80,anchor="center")
        self.pcap_tree.column("c",width=55,anchor="center"); self.pcap_tree.column("b",width=70,anchor="e")
        self.pcap_tree.column("ae",width=60,anchor="center")
        self.pcap_tree.pack(fill="both",expand=True)
        for tag,color in [("strong",self.GREEN),("moderate",self.YELLOW),("weak",self.RED),("critical",self.RED)]:
            self.pcap_tree.tag_configure(tag,foreground=color)
        self.pcap_tree.bind("<<TreeviewSelect>>",self._pcap_sel)
        rf=tk.Frame(pn,bg=self.CARD_BG); pn.add(rf,width=500)
        self.pcap_det=scrolledtext.ScrolledText(rf,wrap="word",bg=self.ENTRY_BG,fg=self.FG,
            font=("Consolas",9),insertbackground=self.FG,bd=0,padx=12,pady=8)
        self.pcap_det.pack(fill="both",expand=True)
        self.pcap_results=[]

    def _run_pcap(self):
        p=self.pcap_path.get()
        if not p or not os.path.exists(p): messagebox.showerror("Error","Select a PCAP file"); return
        if not HAS_SCAPY: messagebox.showerror("Error","pip install scapy"); return
        # Clear previous
        for i in self.pcap_tree.get_children(): self.pcap_tree.delete(i)
        self.pcap_det.delete("1.0","end")
        self.pcap_results=[]
        self.pcap_prog["value"]=0; self.pcap_prog_label.config(text="")
        # Show file size
        fsize=os.path.getsize(p)
        self.pcap_det.insert("end",f"Loading {Path(p).name} ({fsize/1024/1024:.1f} MB)...\n")
        self._set_status(f"Loading PCAP ({fsize/1024/1024:.1f} MB)...")
        threading.Thread(target=self._pcap_work,args=(p,fsize),daemon=True).start()

    def _pcap_prog_update(self, pct, msg):
        """Update progress bar and label from any thread."""
        self.root.after(0, lambda: (
            self.pcap_prog.configure(value=min(pct,100)),
            self.pcap_prog_label.config(text=msg)
        ))

    def _pcap_work(self, path, fsize):
        try:
            self._ensure_trained()
            ps=self.pcap_ports.get().strip()
            ports=None if ps=="all" else set(int(x.strip()) for x in ps.split(",") if x.strip().isdigit())
            ports_set=ports  # Set for fast lookup

            # ── Phase 1: Stream packets using PcapReader (low memory) ──
            streams={}
            stream_bytes={}  # Track total bytes per stream to cap memory
            MAX_STREAM_BYTES=32768  # 32KB cap per stream — enough for analysis
            MIN_STREAM_BYTES=64    # Skip streams smaller than this
            pkt_count=0
            t0=time.time()

            self._pcap_prog_update(0, "Reading packets...")

            try:
                reader=PcapReader(path)
            except Exception:
                # Fallback for compressed/weird formats
                reader=rdpcap(path)

            for pkt in reader:
                pkt_count+=1
                # Progress every 5000 packets
                if pkt_count%5000==0:
                    elapsed=time.time()-t0
                    # Estimate progress from file position if available
                    pct=min(50, int(pkt_count/max(fsize/200,1)*50))  # Rough estimate
                    rate=pkt_count/max(elapsed,0.1)
                    self._pcap_prog_update(pct, f"{pkt_count:,} pkts ({rate:.0f}/s) | {len(streams)} streams")

                if not pkt.haslayer(IP): continue
                ip=pkt[IP]; sp=dp=0; pay=b''
                if pkt.haslayer(TCP): sp,dp=pkt[TCP].sport,pkt[TCP].dport
                elif pkt.haslayer(UDP): sp,dp=pkt[UDP].sport,pkt[UDP].dport
                if pkt.haslayer(Raw): pay=bytes(pkt[Raw].load)
                if not pay: continue
                if ports_set and sp not in ports_set and dp not in ports_set: continue

                k=(ip.src,ip.dst,sp,dp)
                # Cap per-stream memory
                current=stream_bytes.get(k,0)
                if current>=MAX_STREAM_BYTES: continue
                if k not in streams: streams[k]=[]
                keep=pay[:MAX_STREAM_BYTES-current]
                streams[k].append(keep)
                stream_bytes[k]=current+len(keep)

            if hasattr(reader,'close'): reader.close()
            elapsed=time.time()-t0
            self._pcap_prog_update(50, f"Loaded {pkt_count:,} packets in {elapsed:.1f}s | {len(streams)} streams")
            self.root.after(0, lambda: self.pcap_det.insert("end",
                f"Loaded {pkt_count:,} packets in {elapsed:.1f}s\n"
                f"Found {len(streams)} streams with payload data\n\n"
                f"Analyzing streams...\n"))

            # ── Phase 2: Analyze each stream (with progress + incremental display) ──
            results=[]
            stream_data={}  # Keep data for cross-stream attack
            ai_samples={}   # Keep first 256B for AI analysis (Phase 4)
            harvester=self.harvester  # Use persistent harvester — keys accumulate across sessions
            total_streams=len(streams)
            analyzed=0
            alerts=0
            t1=time.time()

            for k,plist in streams.items():
                combined=b''.join(plist)
                if len(combined)<MIN_STREAM_BYTES: continue

                r=self._analyze_stream(k,combined,len(plist),capture_name=Path(path).name)
                results.append(r)
                analyzed+=1
                if r["prediction"]>=2: alerts+=1

                # Only keep data for non-protocol streams (cross-stream can't crack TLS)
                is_protocol = r.get("protocol_override") or any(
                    p.get("method")=="Protocol Detection" for p in r.get("proofs",[]))
                if not is_protocol:
                    stream_data[r["stream"]]=combined
                    ai_samples[r["stream"]]=combined[:256]  # Small sample for AI Phase 4
                # Harvest keys from this stream's proofs
                harvester.harvest(r["stream"], r.get("proofs",[]), capture_name=Path(path).name)

                # Progress
                pct=50+int(analyzed/max(total_streams,1)*45)
                self._pcap_prog_update(pct, f"Stream {analyzed}/{total_streams} | {alerts} alerts")

                # Incremental tree update every stream
                idx=len(results)-1
                tag=r["label"].lower()
                self.root.after(0, lambda i=idx,rr=r,tg=tag: self.pcap_tree.insert("","end",
                    iid=str(i),text=rr["stream"],
                    values=(rr["label"],f"{rr['confidence']:.0%}",f"{rr['bytes']:,}",f"{rr['ae_score']:.2f}"),
                    tags=(tg,)))

            elapsed2=time.time()-t1
            del streams; del stream_bytes  # Free loading memory

            # ── Phase 3: Iterative cross-stream key reuse attack ──
            # Loops until no new discoveries — if cracking stream X reveals a new key,
            # that key gets tried against all remaining uncracked streams
            cross_hits=0
            iteration=0
            MAX_ITERATIONS=5
            if harvester.keys:
                self._pcap_prog_update(95, f"Cross-stream attack ({len(harvester.keys)} keys)...")
                self.root.after(0, lambda: self.pcap_det.insert("end",
                    f"\nCross-stream key attack: {len(harvester.keys)} keys vs {len(results)} streams...\n"))

                while iteration < MAX_ITERATIONS:
                    iteration+=1
                    new_hits=0
                    keys_before=len(harvester.keys)

                    for i,r in enumerate(results):
                        # Skip protocol-encrypted streams — can't XOR-crack TLS
                        if r.get("protocol_override") or any(
                            p.get("method")=="Protocol Detection" for p in r.get("proofs",[])):
                            continue

                        # Only attack streams that weren't already cracked
                        existing_cracks=[p for p in r.get("proofs",[]) if p.get("success") and
                            p.get("method") not in ("Protocol Detection","String Extraction",
                                "Pattern Detection","Entropy Windowing","Known Pattern Match",
                                "DNS Decode","RC4 Bias Detection","Substitution Analysis","AI Traffic Analysis")]
                        if existing_cracks: continue

                        data=stream_data.get(r["stream"],b'')
                        if len(data)<32: continue

                        cross_proofs=harvester.attack_stream(r["stream"], data)
                        if cross_proofs:
                            new_hits+=1; cross_hits+=1
                            r["proofs"].extend(cross_proofs)
                            r["label"]="WEAK"; r["prediction"]=2
                            r["cracked_override"]=True
                            r["cross_stream_hit"]=True
                            alerts+=1

                            # Harvest keys and intel from the newly cracked stream
                            harvester.harvest(r["stream"], cross_proofs, capture_name=Path(path).name)
                            self.intel.harvest_from_proofs(r["stream"], data, cross_proofs, capture=Path(path).name)

                            # Update tree
                            self.root.after(0, lambda ii=str(i),rr=r: (
                                self.pcap_tree.item(ii,values=(rr["label"],f"{rr['confidence']:.0%}",
                                    f"{rr['bytes']:,}",f"{rr['ae_score']:.2f}"),tags=("weak",))
                                if self.pcap_tree.exists(ii) else None))

                    keys_after=len(harvester.keys)
                    log.info(f"Cross-stream pass {iteration}: {new_hits} hits, {keys_after-keys_before} new keys")

                    if new_hits==0 or keys_after==keys_before:
                        break  # No new discoveries, stop iterating

                    self._pcap_prog_update(96, f"Cross-stream pass {iteration+1} ({keys_after} keys)...")

                if iteration>1:
                    log.info(f"Cross-stream completed in {iteration} passes, {cross_hits} total hits")

            elapsed3=time.time()-t1-elapsed2
            del stream_data  # Free cross-stream memory
            reuse_report=harvester.get_reuse_report()

            # ── Phase 4: Parallel AI analysis (if Ollama available) ──
            ai_hits=0
            elapsed4=0
            if self.settings.get("auto_validate") and self.validator._check():
                from concurrent.futures import ThreadPoolExecutor, as_completed
                # Only analyze interesting non-protocol streams
                ai_candidates=[]
                for r in results:
                    if r.get("protocol_override"): continue
                    if any(p.get("method")=="Protocol Detection" for p in r.get("proofs",[])): continue
                    if any(p.get("method")=="Plaintext Protocol" for p in r.get("proofs",[])): continue
                    if any(p.get("method")=="DNS Decode" for p in r.get("proofs",[])): continue
                    if r["prediction"] >= 1 and r["bytes"] >= 64:
                        ai_candidates.append(r)
                ai_candidates=ai_candidates[:20]  # Cap at 20

                if ai_candidates:
                    self._pcap_prog_update(96, f"AI analyzing {len(ai_candidates)} streams...")
                    t4=time.time()

                    def _ai_work(r):
                        try:
                            data_sample = ai_samples.get(r["stream"], b'')
                            if not data_sample: return (r, None)
                            ai_r = self.validator.analyze_traffic(
                                r["stream"], data_sample, r, r.get("proofs",[]), r.get("protocol",{}))
                            return (r, ai_r)
                        except: return (r, None)

                    # Run 4 threads in parallel
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        futures={pool.submit(_ai_work, r): r for r in ai_candidates}
                        for fut in as_completed(futures, timeout=120):
                            try:
                                r, ai_r = fut.result(timeout=30)
                                if ai_r:
                                    r["proofs"].append(ai_r)
                                    ai_hits+=1
                            except: pass
                    elapsed4=time.time()-t4
            del ai_samples  # Free AI sample memory

            self.pcap_results=results; self._session_results.extend(results)

            # ── Final summary ──
            total_time=elapsed+elapsed2+elapsed3+elapsed4
            self._pcap_prog_update(100, f"Done: {analyzed} streams in {total_time:.1f}s")
            by=collections.Counter(r["label"] for r in results)

            def finish():
                self.pcap_det.delete("1.0","end")
                self.pcap_det.insert("end",f"═══ Analysis Complete ═══\n\n")
                self.pcap_det.insert("end",f"  File:     {Path(path).name} ({fsize/1024/1024:.1f} MB)\n")
                self.pcap_det.insert("end",f"  Packets:  {pkt_count:,}\n")
                self.pcap_det.insert("end",f"  Streams:  {analyzed} analyzed ({total_streams-analyzed} skipped < {MIN_STREAM_BYTES}B)\n")
                self.pcap_det.insert("end",f"  Time:     {total_time:.1f}s "
                    f"(load: {elapsed:.1f}s, analyze: {elapsed2:.1f}s, cross-stream: {elapsed3:.1f}s"
                    f"{f', AI: {elapsed4:.1f}s ({ai_hits} analyzed)' if elapsed4>0 else ''}"
                    f"{f', {iteration} passes' if iteration>1 else ''})\n\n")
                for lb in ["CRITICAL","WEAK","MODERATE","STRONG"]:
                    if lb in by: self.pcap_det.insert("end",f"  {lb}: {by[lb]} stream(s)\n")

                # Key harvest summary
                rr=reuse_report
                self.pcap_det.insert("end",f"\n  ── Key Intelligence ──\n")
                self.pcap_det.insert("end",f"  Keys in vault:   {rr['total_keys_harvested']} (persisted across sessions)\n")
                self.pcap_det.insert("end",f"  Unique keys:     {rr['unique_keys']}\n")
                if rr['keys_reused_across_streams']>0:
                    self.pcap_det.insert("end",f"  ⚠ KEY REUSE:     {rr['keys_reused_across_streams']} keys used across multiple streams\n")
                    for kh,info in rr['reuse_details'].items():
                        self.pcap_det.insert("end",f"    Key {kh[:16]}... → {info['count']} streams\n")
                        for s in info['streams'][:3]:
                            self.pcap_det.insert("end",f"      • {s}\n")
                if rr.get('cross_capture_reuse'):
                    self.pcap_det.insert("end",f"\n  🔴 CROSS-CAPTURE REUSE: {len(rr['cross_capture_reuse'])} keys found in multiple captures!\n")
                    for kh,info in rr['cross_capture_reuse'].items():
                        self.pcap_det.insert("end",f"    Key {kh[:16]}... → captures: {', '.join(info['captures'])}\n")
                if rr.get('prefix_clusters'):
                    self.pcap_det.insert("end",f"\n  🔍 KEY CLUSTERS: {len(rr['prefix_clusters'])} groups sharing same 4-byte prefix\n")
                    for prefix,info in sorted(rr['prefix_clusters'].items(), key=lambda x:-x[1]['count'])[:5]:
                        self.pcap_det.insert("end",f"    Prefix {prefix}... × {info['count']} streams\n")
                        for s in info['streams'][:3]:
                            self.pcap_det.insert("end",f"      • {s}\n")
                        if info.get('cross_capture'):
                            self.pcap_det.insert("end",f"      ⚠ Seen across: {', '.join(info['captures'])}\n")
                if cross_hits>0:
                    self.pcap_det.insert("end",f"  🔓 CROSS-STREAM:  {cross_hits} additional streams decrypted!\n")
                else:
                    self.pcap_det.insert("end",f"  Cross-stream:    No additional decryptions found\n")

                # Intel summary
                intel_r=self.intel.get_report()
                if intel_r["total_credentials"]>0 or intel_r["total_cribs"]>0:
                    self.pcap_det.insert("end",f"\n  ── Intel Collected ──\n")
                    self.pcap_det.insert("end",f"  Credentials:     {intel_r['total_credentials']}")
                    if intel_r["credential_types"]:
                        self.pcap_det.insert("end",f" ({', '.join(f'{v}×{k}' for k,v in intel_r['credential_types'].items())})")
                    self.pcap_det.insert("end","\n")
                    self.pcap_det.insert("end",f"  Auto-cribs:      {intel_r['total_cribs']} (will test on future captures)\n")
                    self.pcap_det.insert("end",f"  Patterns stored: {intel_r['total_patterns']}\n")
                    if intel_r.get("cross_capture_credentials"):
                        self.pcap_det.insert("end",f"  🔴 CROSS-CAPTURE CREDS: {len(intel_r['cross_capture_credentials'])} seen in multiple captures\n")
                        for cred,caps in list(intel_r["cross_capture_credentials"].items())[:5]:
                            self.pcap_det.insert("end",f"    {cred} → {', '.join(caps)}\n")

                self.pcap_det.insert("end",f"\n  Click a stream on the left to see full details.\n")
                self._set_status(f"Done: {analyzed} streams ({alerts} alerts, {cross_hits} cross-stream) in {elapsed+elapsed2+elapsed3:.1f}s")
                self.c_sess.config(text=f"{len(self._session_results)} results")
                self._refresh_dashboard()
                self._maybe_auto_train()  # Auto-retrain if enough new samples
            self.root.after(0,finish)

        except Exception as e:
            log.error(f"PCAP error: {traceback.format_exc()}")
            self.root.after(0,lambda:self.pcap_det.insert("end",f"\nError: {e}"))
            self.root.after(0,lambda:self._pcap_prog_update(0,"Error"))

    def _analyze_stream(self,key,data,npkt,capture_name=""):
        src,dst,sp,dp=key; sid=f"{src}:{sp} → {dst}:{dp}"
        ml=self.models.predict(data); pan=PANDetector.scan(data)
        if pan: ml["label"]="CRITICAL"; ml["prediction"]=3

        # Protocol-aware classification override
        proto=ProtocolDetector.detect(data, sp, dp)

        # DNS by port — decode directly, skip encryption analysis
        is_dns = sp in (53,853) or dp in (53,853)
        if is_dns:
            proto={"protocol":"DNS","encrypted_properly":False}
            ml["protocol"]="DNS"
            # Try DNS decode
            dns_result=DNSDecoder.decode(data)
            if dns_result and dns_result.get("success"):
                # Validate: domain names must be printable ASCII
                valid_domains=[d for d in dns_result.get("query_names",[])
                               if d and all(32<=ord(c)<=126 for c in d)]
                if valid_domains:
                    ml["label"]="MODERATE"; ml["prediction"]=1
                    port_label=PORT_MAP.get(dp,PORT_MAP.get(sp,""))
                    return {"stream":sid,"port":port_label,"bytes":len(data),"packets":npkt,
                            "pan":pan,"proofs":[dns_result],"validations":[],"protocol":proto,
                            "auto_cribs_available":0,**ml}
            # DNS port but couldn't decode — fall through to normal analysis
            if ml["prediction"]==0:
                ml["label"]="MODERATE"; ml["prediction"]=1

        # Plaintext protocol detection — HTTP, FTP, Telnet, SNMP, MQTT, Modbus, SIP, etc.
        plaintext_result = PlaintextProtocolAnalyzer.analyze(data, sp, dp)
        if plaintext_result and plaintext_result.get("success"):
            proto = {"protocol": plaintext_result["protocol"], "encrypted_properly": False}
            ml["protocol"] = plaintext_result["protocol"]
            ml["label"] = "CRITICAL" if plaintext_result.get("credentials") else "WEAK"
            ml["prediction"] = 3 if plaintext_result.get("credentials") else 2
            # Extract credentials into intel
            for cred in plaintext_result.get("credentials", []):
                ctype = cred.get("type", "CREDENTIAL")
                cval = cred.get("username", cred.get("value", cred.get("user", "")))
                if cval:
                    self.intel.credentials.append({"type": ctype, "value": cval,
                        "source": sid, "capture": capture_name,
                        "method": "Plaintext Protocol", "time": datetime.now().isoformat()})
            port_label = PORT_MAP.get(dp, PORT_MAP.get(sp, ""))
            return {"stream": sid, "port": port_label, "bytes": len(data), "packets": npkt,
                    "pan": pan, "proofs": [plaintext_result], "validations": [], "protocol": proto,
                    "auto_cribs_available": 0, **ml}

        # TLS handshake parsing — extract SNI, certs, cipher suites from unencrypted handshake
        tls_hs = TLSHandshakeParser.parse(data)

        if proto.get("encrypted_properly") and ml["prediction"]>=2 and not pan:
            ml["label"]="STRONG"; ml["prediction"]=0
            ml["protocol_override"]=True
            ml["protocol"]=proto.get("protocol","")
            ml["protocol_version"]=proto.get("version","")
            if proto.get("weak_version"):
                ml["label"]="MODERATE"; ml["prediction"]=1

        # Check for known encrypted patterns from previous sessions
        pattern_hits=self.intel.check_pattern(data)

        # Auto-cribs — only on non-protocol streams (don't crib-attack TLS)
        auto_cribs=[]
        best_crib=None
        if not proto.get("encrypted_properly"):
            auto_cribs=self.intel.get_auto_cribs()
            best_crib=max(auto_cribs, key=len) if auto_cribs else None

        # Run decryption with auto-crib if available
        proofs=DecryptionEngine.attempt_all(data,ml,known_plaintext=best_crib) if self.settings.get("auto_decrypt",True) else []

        # Validate auto-crib results — require high printability to avoid false positives
        validated_proofs=[]
        for p in proofs:
            if p.get("method")=="Known-Plaintext Attack" and p.get("success"):
                pr=p.get("printable",0)
                preview=p.get("preview","")
                # Auto-cribs need stricter validation than user-provided cribs
                if pr<0.7 or DecryptionEngine._is_garbled(preview):
                    continue  # Skip this garbled result
            validated_proofs.append(p)
        proofs=validated_proofs

        # If the first auto-crib didn't produce a quality hit, try others (up to 3)
        if best_crib and not any(p.get("method")=="Known-Plaintext Attack" and p.get("success")
                                  and p.get("printable",0)>0.7 for p in proofs):
            for crib in sorted(auto_cribs, key=len, reverse=True)[1:3]:
                if len(crib)<8: continue
                extra=DecryptionEngine._known_plaintext(data, crib)
                if extra and extra.get("success") and extra.get("printable",0)>0.7:
                    extra["auto_crib"]=True
                    extra["crib_source"]="intel_vault"
                    proofs.append(extra)
                    break

        # Add pattern match findings
        proofs.extend(pattern_hits)

        # Add TLS handshake intelligence if found
        if tls_hs and tls_hs.get("success"):
            proofs.append(tls_hs)

        vals=[]
        if proofs and self.settings.get("auto_validate"):
            for pr in proofs:
                if pr.get("success") and pr.get("method") not in (
                    "Protocol Detection","Known Pattern Match","TLS Handshake",
                    "Plaintext Protocol","DNS Decode","AI Traffic Analysis"):
                    v=self.validator.validate(pr,data); vals.append(v)
                    self.feedback.record(data,ml,pr,v)

        # Harvest intel from successful proofs
        self.intel.harvest_from_proofs(sid, data, proofs, capture=capture_name)

        # If decryption found real weaknesses, upgrade severity
        real_cracks=[p for p in proofs if p.get("success") and p.get("method") not in
            ("String Extraction","Entropy Windowing","Protocol Detection","Pattern Detection",
             "Known Pattern Match","DNS Decode","RC4 Bias Detection","Substitution Analysis",
             "AI Traffic Analysis","TLS Handshake","Plaintext Protocol")
            and p.get("confidence",0)>0.5 and not DecryptionEngine._is_garbled(p.get("preview",""))]
        if real_cracks and ml["prediction"]==0:
            ml["label"]="WEAK"; ml["prediction"]=2
            ml["cracked_override"]=True

        # AI traffic analysis is deferred to Phase 4 (parallel batch)
        # to avoid blocking the analysis loop with sequential Ollama calls

        port_label=PORT_MAP.get(dp,PORT_MAP.get(sp,""))
        return {"stream":sid,"port":port_label,"bytes":len(data),"packets":npkt,
                "pan":pan,"proofs":proofs,"validations":vals,"protocol":proto,
                "auto_cribs_available":len(auto_cribs),**ml}

    def _pcap_show(self):
        """Legacy — results now shown incrementally during analysis."""
        pass

    def _pcap_sel(self,_):
        sel=self.pcap_tree.selection()
        if sel: self._show_detail(self.pcap_results[int(sel[0])],self.pcap_det)

    # ── Live Capture ──
    def _tab_live(self):
        t=ttk.Frame(self.nb); self.nb.add(t,text="  Live Capture  ")
        ctrl=tk.Frame(t,bg=self.BG); ctrl.pack(fill="x",padx=16,pady=12)
        tk.Label(ctrl,text="Interface:",fg=self.FG,bg=self.BG).pack(side="left")
        self.cap_iface=tk.StringVar(value=self.settings.get("capture_iface",""))
        ttk.Entry(ctrl,textvariable=self.cap_iface,width=15).pack(side="left",padx=4)
        tk.Label(ctrl,text="BPF:",fg=self.FG,bg=self.BG).pack(side="left",padx=(8,4))
        self.cap_bpf=tk.StringVar(value=self.settings.get("capture_bpf",DEFAULT_BPF))
        ttk.Entry(ctrl,textvariable=self.cap_bpf,width=50).pack(side="left",padx=4)
        self.cap_btn=ttk.Button(ctrl,text="▶ Start",style="Accent.TButton",command=self._toggle_live)
        self.cap_btn.pack(side="right",padx=8)

        sb=tk.Frame(t,bg=self.BG2,height=28); sb.pack(fill="x",padx=16); sb.pack_propagate(False)
        self.cap_stats=tk.Label(sb,text="Packets: 0 | Streams: 0 | Alerts: 0",
            fg=self.FG2,bg=self.BG2,font=("Consolas",9))
        self.cap_stats.pack(side="left",padx=8)
        self.cap_ind=tk.Label(sb,text="● STOPPED",fg=self.FG2,bg=self.BG2,font=("Segoe UI",9,"bold"))
        self.cap_ind.pack(side="right",padx=8)

        pn=tk.PanedWindow(t,orient="horizontal",bg=self.BG,sashwidth=4)
        pn.pack(fill="both",expand=True,padx=16,pady=(8,12))
        lf=tk.Frame(pn,bg=self.CARD_BG); pn.add(lf,width=520)
        self.live_tree=ttk.Treeview(lf,columns=("v","c","b","pk"),show="tree headings")
        self.live_tree.heading("#0",text="Stream"); self.live_tree.heading("v",text="Verdict")
        self.live_tree.heading("c",text="Conf"); self.live_tree.heading("b",text="Bytes")
        self.live_tree.heading("pk",text="Pkts")
        self.live_tree.column("#0",width=200); self.live_tree.column("v",width=80,anchor="center")
        self.live_tree.column("c",width=55,anchor="center"); self.live_tree.column("b",width=70,anchor="e")
        self.live_tree.column("pk",width=50,anchor="center")
        self.live_tree.pack(fill="both",expand=True)
        for tag,color in [("strong",self.GREEN),("moderate",self.YELLOW),("weak",self.RED),("critical",self.RED)]:
            self.live_tree.tag_configure(tag,foreground=color)
        self.live_tree.bind("<<TreeviewSelect>>",self._live_sel)
        rf=tk.Frame(pn,bg=self.CARD_BG); pn.add(rf,width=500)
        self.live_det=scrolledtext.ScrolledText(rf,wrap="word",bg=self.ENTRY_BG,fg=self.FG,
            font=("Consolas",9),insertbackground=self.FG,bd=0,padx=12,pady=8)
        self.live_det.pack(fill="both",expand=True)
        self._lstreams={}; self._lresults={}; self._lpkts=0; self._lalerts=0; self._lstream_sizes={}

    def _toggle_live(self):
        if self._live_on: self._live_on=False; self.cap_btn.config(text="▶ Start"); self.cap_ind.config(text="● STOPPED",fg=self.FG2)
        else:
            if not HAS_SCAPY: messagebox.showerror("Error","pip install scapy"); return
            self._live_on=True; self._lstreams={}; self._lresults={}; self._lpkts=0; self._lalerts=0
            self._lstream_sizes={}; self._auto_train_counter=0
            self.cap_btn.config(text="■ Stop"); self.cap_ind.config(text="● CAPTURING",fg=self.GREEN)
            self.settings["capture_iface"]=self.cap_iface.get()
            self.settings["capture_bpf"]=self.cap_bpf.get(); self._save_settings()
            threading.Thread(target=self._live_sniff,daemon=True).start()
            self._live_timer()

    def _live_sniff(self):
        try:
            from scapy.sendrecv import sniff as ssniff
        except: from scapy.all import sniff as ssniff
        iface=self.cap_iface.get().strip() or None; bpf=self.cap_bpf.get().strip() or None
        try:
            ssniff(iface=iface,filter=bpf,prn=self._live_pkt,store=False,stop_filter=lambda p:not self._live_on)
        except Exception as e:
            log.error(f"Capture error: {e}")
            self.root.after(0,lambda:messagebox.showerror("Capture Error",f"{e}\n\nRun as Administrator."))
            self.root.after(0,lambda:(setattr(self,'_live_on',False),self.cap_btn.config(text="▶ Start")))

    def _live_pkt(self,pkt):
        if not pkt.haslayer(IP): return
        ip=pkt[IP]; sp=dp=0; pay=b''
        if pkt.haslayer(TCP): sp,dp=pkt[TCP].sport,pkt[TCP].dport
        elif pkt.haslayer(UDP): sp,dp=pkt[UDP].sport,pkt[UDP].dport
        if pkt.haslayer(Raw): pay=bytes(pkt[Raw].load)
        if not pay: return
        self._lpkts+=1; k=(ip.src,ip.dst,sp,dp)
        # Cap per-stream memory at 32KB
        current_size=self._lstream_sizes.get(k,0)
        if current_size>=32768: return
        keep=pay[:32768-current_size]
        self._lstreams.setdefault(k,[]).append(keep)
        self._lstream_sizes[k]=current_size+len(keep)

    def _live_timer(self):
        if not self._live_on: return
        mem_mb=sum(self._lstream_sizes.values())/1024/1024
        self.cap_stats.config(text=f"Pkts: {self._lpkts} | Streams: {len(self._lstreams)} | "
            f"Alerts: {self._lalerts} | Keys: {len(self.harvester.keys)} | RAM: {mem_mb:.0f}MB")
        if self.models.is_trained:
            threading.Thread(target=self._live_analyze,daemon=True).start()
        # Auto-retrain every 200 new samples
        if hasattr(self,'_auto_train_counter'):
            self._auto_train_counter+=1
            if self._auto_train_counter>=60:  # Every 3 min check
                self._auto_train_counter=0
                self._maybe_auto_train()
        self.root.after(3000,self._live_timer)

    def _live_analyze(self):
        analyzed_this_round=0
        for k,plist in list(self._lstreams.items()):
            if analyzed_this_round>=20: break  # Limit per cycle to stay responsive
            combined=b''.join(plist)
            if len(combined)<64: continue
            prev=self._lresults.get(k)
            if prev and prev.get("bytes",0)>len(combined)*0.9: continue
            try:
                r=self._analyze_stream(k,combined,len(plist),capture_name="live_capture")
                analyzed_this_round+=1

                # Harvest keys
                self.harvester.harvest(r["stream"], r.get("proofs",[]), capture_name="live_capture")

                # Cross-stream — skip protocol-encrypted
                is_protocol = r.get("protocol_override") or any(
                    p.get("method")=="Protocol Detection" for p in r.get("proofs",[]))
                existing_cracks=[p for p in r.get("proofs",[]) if p.get("success") and
                    p.get("method") not in ("Protocol Detection","String Extraction","Pattern Detection",
                        "Entropy Windowing","DNS Decode","RC4 Bias Detection","Substitution Analysis","AI Traffic Analysis")]
                if not is_protocol and not existing_cracks and self.harvester.keys:
                    cross_proofs=self.harvester.attack_stream(r["stream"], combined)
                    if cross_proofs:
                        r["proofs"].extend(cross_proofs)
                        r["label"]="WEAK"; r["prediction"]=2
                        r["cracked_override"]=True; r["cross_stream_hit"]=True
                        self._lalerts+=1

                if r["prediction"]>=2 and k not in self._lresults: self._lalerts+=1
                self._lresults[k]=r

                # Only add to session results if interesting (save memory)
                if r["prediction"]>=1:  # MODERATE or worse
                    self._session_results.append(r)

                # Evict raw payload data after analysis for STRONG streams
                if r["prediction"]==0 and k in self._lstreams:
                    freed=self._lstream_sizes.get(k,0)
                    self._lstreams[k]=[b'']  # Keep key but free payload
                    self._lstream_sizes[k]=0

                # Only add to TreeView if interesting or first 100
                tree_count=len(self.live_tree.get_children())
                iid=str(hash(k)); tag=r["label"].lower()
                vals=(r["label"],f"{r['confidence']:.0%}",f"{r['bytes']:,}",r.get("packets","?"))
                if r["prediction"]>=2 or tree_count<100:
                    self.root.after(0,lambda ii=iid,s=r["stream"],v=vals,tg=tag:
                        self.live_tree.item(ii,text=s,values=v,tags=(tg,)) if self.live_tree.exists(ii) else
                        self.live_tree.insert("","end",iid=ii,text=s,values=v,tags=(tg,)))
                elif self.live_tree.exists(iid):
                    self.root.after(0,lambda ii=iid,v=vals,tg=tag:
                        self.live_tree.item(ii,values=v,tags=(tg,)))
            except: pass

    def _live_sel(self,_):
        sel=self.live_tree.selection()
        if not sel: return
        iid=sel[0]
        for k,r in self._lresults.items():
            if str(hash(k))==iid: self._show_detail(r,self.live_det); break

    # ── Payload Analysis ──
    def _tab_payload(self):
        t=ttk.Frame(self.nb); self.nb.add(t,text="  Payload Analysis  ")

        # Top input section
        top=tk.Frame(t,bg=self.BG); top.pack(fill="x",padx=16,pady=(12,0))
        inf=tk.LabelFrame(top,text=" Input — Paste anything or load any file ",
            bg=self.CARD_BG,fg=self.ACCENT,font=("Segoe UI",10,"bold"),bd=1,relief="solid")
        inf.pack(fill="x")

        # Button row
        br=tk.Frame(inf,bg=self.CARD_BG); br.pack(fill="x",padx=8,pady=(8,4))
        ttk.Button(br,text="📁 Load File",command=self._load_file).pack(side="left",padx=4)
        ttk.Button(br,text="🎲 Demo",command=self._demo_payload).pack(side="left",padx=4)
        ttk.Button(br,text="🗑 Clear",command=lambda:(
            self.pl_input.delete("1.0","end"),setattr(self,'pl_data',None),
            self.pl_crib.delete(0,"end"))).pack(side="left",padx=4)
        ttk.Button(br,text="▶ Analyze",style="Accent.TButton",command=self._analyze_pl).pack(side="right",padx=4)
        # Format hint
        tk.Label(br,text="Accepts: text, hex, base64, binary, .json, .csv, .txt, .enc, .dat, .bin, .pcap",
            fg=self.FG2,bg=self.CARD_BG,font=("Segoe UI",7)).pack(side="right",padx=8)

        # Main input area (bigger)
        self.pl_input=scrolledtext.ScrolledText(inf,height=6,wrap="word",bg=self.ENTRY_BG,fg=self.FG,
            font=("Consolas",10),insertbackground=self.FG,bd=0,padx=8,pady=6)
        self.pl_input.pack(fill="x",padx=8,pady=(0,4))
        self.pl_input.insert("1.0","Paste encrypted text, hex, base64, or any data here.\n"
            "Or load a file — any format works.\n\n"
            "Examples:\n"
            "  • Raw text: Gur dhvpx oebja sbk (ROT13)\n"
            "  • Hex: 48 65 6c 6c 6f 20 57 6f 72 6c 64\n"
            "  • Base64: SGVsbG8gV29ybGQ=\n"
            "  • Just paste encrypted output from anything")

        # Known plaintext row (for crib attacks)
        crib_row=tk.Frame(inf,bg=self.CARD_BG); crib_row.pack(fill="x",padx=8,pady=(0,8))
        tk.Label(crib_row,text="Known plaintext (optional):",fg=self.FG2,bg=self.CARD_BG,
            font=("Segoe UI",9)).pack(side="left")
        self.pl_crib=ttk.Entry(crib_row,width=50)
        self.pl_crib.pack(side="left",padx=4)
        tk.Label(crib_row,text="If you know part of the original text, enter it here for crib attack",
            fg=self.FG2,bg=self.CARD_BG,font=("Segoe UI",7)).pack(side="left",padx=4)

        self.pl_data=None

        # Results
        self.pl_out=scrolledtext.ScrolledText(t,wrap="word",bg=self.ENTRY_BG,fg=self.FG,
            font=("Consolas",9),insertbackground=self.FG,bd=0,padx=12,pady=8)
        self.pl_out.pack(fill="both",expand=True,padx=16,pady=(8,12))

    def _load_file(self):
        f=filedialog.askopenfilename(filetypes=[
            ("All supported","*.bin *.raw *.hex *.txt *.enc *.encrypted *.cipher *.dat "
             "*.json *.csv *.log *.key *.pcap *.pcapng *.cap"),("All files","*.*")])
        if not f: return
        data,desc=InputParser.from_file(f)
        if desc=="pcap":
            self.pcap_path.set(f); self.nb.select(1); return
        self.pl_data=data
        self.pl_input.delete("1.0","end")
        preview_hex=data[:80].hex() if data else 'empty'
        preview_ascii=''.join(chr(b) if 32<=b<=126 else '·' for b in (data or b'')[:80])
        self.pl_input.insert("1.0",
            f"Loaded: {Path(f).name}\n"
            f"Format: {desc}\n"
            f"Hex:    {preview_hex}\n"
            f"ASCII:  {preview_ascii}")
        log.info(f"Loaded: {f} ({desc})")

    def _demo_payload(self):
        demos = collections.OrderedDict([
            ("XOR encrypted card data",lambda:bytes(
                p^b'\xAB\xCD\xEF\x12\x34\x56\x78\x9A'[i%8] for i,p in enumerate(
                b"CARD:4532015112830366 EXP:12/27 CVV:123 AMT:$499.99 "*20))),
            ("ROT13 text",lambda:bytes(
                ((b-65+13)%26+65 if 65<=b<=90 else (b-97+13)%26+97 if 97<=b<=122 else b)
                for b in b"The quick brown fox jumps over the lazy dog. Card number is hidden.")),
            ("Vigenère cipher",lambda:bytes(
                (b-65+k)%26+65 if 65<=b<=90 else (b-97+k)%26+97 if 97<=b<=122 else b
                for b,k in zip(b"Attack at dawn with the secret key and the encrypted message repeated many times for analysis",
                    [ord(c)-65 for c in "SECRET"*100]))),
            ("Base64 encoded secrets",lambda:base64.b64encode(
                b"password=Admin123&api_key=sk_live_abc123def456&card=4111111111111111"*5)),
            ("Multi-layer: b64(hex(text))",lambda:base64.b64encode(
                b"card_number=4532015112830366&cvv=123".hex().encode())),
            ("Custom XOR pattern",lambda:bytes(
                p^(0x42+(i*3)%256)%256 for i,p in enumerate(
                b"SECRET_DATA:username=admin,password=hunter2,token=eyJhbG..."*10))),
            ("ECB repeated blocks",lambda:(os.urandom(16)*30+os.urandom(16)*5+os.urandom(16)*20)),
            ("Plaintext card transaction",lambda:
                b"POST /process HTTP/1.1\r\nContent-Type: application/json\r\n\r\n"
                b'{"card":"4532015112830366","exp":"12/27","cvv":"123","amount":499.99}'),
            ("Strong encryption (random)",lambda:os.urandom(2048)),
        ])
        win=tk.Toplevel(self.root); win.title("Demo Payloads"); win.geometry("380x330"); win.configure(bg=self.BG)
        tk.Label(win,text="Choose a test payload:",font=("Segoe UI",11),fg=self.ACCENT,bg=self.BG).pack(pady=8)
        for name,fn in demos.items():
            ttk.Button(win,text=name,command=lambda n=name,f=fn:(
                setattr(self,'pl_data',f()),self.pl_input.delete("1.0","end"),
                self.pl_input.insert("1.0",f"Demo: {n}\n({len(self.pl_data):,} bytes)\n\n"
                    f"Hex: {self.pl_data[:64].hex()}\n"
                    f"ASCII: {''.join(chr(b) if 32<=b<=126 else '·' for b in self.pl_data[:64])}"),
                win.destroy())).pack(fill="x",padx=16,pady=1)

    def _analyze_pl(self):
        # Auto-parse input if no data loaded
        if self.pl_data is None:
            text=self.pl_input.get("1.0","end").strip()
            if not text or text.startswith("Paste encrypted") or text.startswith("Loaded:"):
                messagebox.showinfo("Input needed","Paste text, hex, or base64 into the input box,\n"
                    "or load a file, or choose a demo."); return
            self.pl_data=InputParser.parse_text(text)
        if not self.pl_data or len(self.pl_data)<4:
            messagebox.showerror("Error","No valid data to analyze"); return
        # Get known plaintext crib if provided
        crib_text=self.pl_crib.get().strip()
        self._pl_crib_bytes=crib_text.encode('utf-8') if crib_text else None
        self._set_status("Analyzing...")
        self.pl_out.delete("1.0","end"); self.pl_out.insert("end","Analyzing...\n")
        threading.Thread(target=self._pl_work,daemon=True).start()

    def _pl_work(self):
        try:
            self._ensure_trained()
            data=self.pl_data
            ml=self.models.predict(data); pan=PANDetector.scan(data)
            if pan: ml["label"]="CRITICAL"; ml["prediction"]=3

            # Use user-provided crib, OR auto-cribs from intel vault
            crib=self._pl_crib_bytes
            if not crib:
                auto_cribs=self.intel.get_auto_cribs()
                crib=max(auto_cribs, key=len) if auto_cribs else None

            # Always try decryption — user uploaded it to test
            proofs=DecryptionEngine.attempt_all(data,ml,known_plaintext=crib)

            # Validate auto-crib results (user-provided cribs are trusted)
            if not self._pl_crib_bytes:
                proofs=[p for p in proofs if not (
                    p.get("method")=="Known-Plaintext Attack" and p.get("success")
                    and (p.get("printable",0)<0.7 or DecryptionEngine._is_garbled(p.get("preview",""))))]

            # If user crib didn't hit, try auto-cribs from intel (strict threshold)
            if not self._pl_crib_bytes and not any(p.get("method")=="Known-Plaintext Attack" and p.get("success") for p in proofs):
                for ac in sorted(self.intel.get_auto_cribs(), key=len, reverse=True)[:3]:
                    if len(ac)<8: continue
                    extra=DecryptionEngine._known_plaintext(data, ac)
                    if extra and extra.get("success") and extra.get("printable",0)>0.7:
                        extra["auto_crib"]=True; extra["crib_source"]="intel_vault"
                        proofs.append(extra); break

            # Check for known encrypted patterns
            pattern_hits=self.intel.check_pattern(data)
            proofs.extend(pattern_hits)

            # Harvest keys
            self.harvester.harvest("Payload", proofs, capture_name="payload_analysis")

            # Harvest intel (credentials, cribs, patterns)
            self.intel.harvest_from_proofs("Payload", data, proofs, capture="payload_analysis")

            # Cross-stream key attack
            if self.harvester.keys:
                existing_cracks=[p for p in proofs if p.get("success") and
                    p.get("method") not in ("Protocol Detection","String Extraction","Pattern Detection",
                        "Entropy Windowing","Known Pattern Match","DNS Decode",
                        "RC4 Bias Detection","Substitution Analysis","AI Traffic Analysis")]
                if not existing_cracks:
                    cross_proofs=self.harvester.attack_stream("Payload", data)
                    if cross_proofs:
                        proofs.extend(cross_proofs)
                        ml["label"]="WEAK"; ml["prediction"]=2
                        ml["cross_stream_hit"]=True

            vals=[]
            if proofs and self.settings.get("auto_validate"):
                for pr in proofs:
                    if pr.get("success") and pr.get("method") not in (
                        "Protocol Detection","Known Pattern Match","TLS Handshake",
                        "Plaintext Protocol","DNS Decode","AI Traffic Analysis"):
                        v=self.validator.validate(pr,data); vals.append(v)
                        self.feedback.record(data,ml,pr,v)

            intel_r=self.intel.get_report()
            r={"stream":"Payload","bytes":len(data),"packets":1,"pan":pan,
               "proofs":proofs,"validations":vals,
               "keys_available":len(self.harvester.keys),
               "intel_cribs":intel_r["total_cribs"],
               "intel_creds":intel_r["total_credentials"],**ml}
            self._session_results.append(r)
            self.root.after(0,lambda:self._show_detail(r,self.pl_out))
            self.root.after(0,lambda:self._set_status(
                f"Result: {r['label']} ({r['confidence']:.1%}) | "
                f"{len(self.harvester.keys)} keys, {intel_r['total_cribs']} cribs in vault"))
            self.root.after(0,lambda:(self.c_sess.config(text=f"{len(self._session_results)} results"),
                self._refresh_dashboard()))
        except Exception as e:
            log.error(traceback.format_exc())
            self.root.after(0,lambda:self.pl_out.insert("end",f"\nError: {e}"))

    # ── Training ──
    def _tab_train(self):
        t=ttk.Frame(self.nb); self.nb.add(t,text="  Training  ")
        ctrl=tk.Frame(t,bg=self.BG); ctrl.pack(fill="x",padx=16,pady=12)
        tk.Label(ctrl,text="Samples:",fg=self.FG,bg=self.BG).pack(side="left")
        self.tr_n=tk.IntVar(value=self.settings.get("samples",500))
        ttk.Entry(ctrl,textvariable=self.tr_n,width=6).pack(side="left",padx=4)
        ttk.Button(ctrl,text="▶ Train Synthetic",style="Accent.TButton",command=self._train_syn).pack(side="left",padx=8)
        ttk.Button(ctrl,text="📁 Train from PCAP",command=self._train_pcap).pack(side="left",padx=4)
        ttk.Button(ctrl,text="🔄 Retrain Feedback",command=self._retrain_fb).pack(side="left",padx=4)
        self.tr_prog=ttk.Progressbar(t,mode="indeterminate"); self.tr_prog.pack(fill="x",padx=16,pady=4)
        self.tr_log=scrolledtext.ScrolledText(t,wrap="word",bg=self.ENTRY_BG,fg=self.FG,
            font=("Consolas",9),insertbackground=self.FG,bd=0,padx=12,pady=8)
        self.tr_log.pack(fill="both",expand=True,padx=16,pady=(0,12))
        self.tr_log.insert("end","Models auto-train on first use.\n\n"
            "Train Synthetic — fresh labeled training data\n"
            "Train from PCAP — real streams from your captures\n"
            "Retrain Feedback — learn from previous analysis\n")

    def _tlog(self,msg):
        self.root.after(0,lambda m=msg:(self.tr_log.insert("end",f"  {m}\n"),self.tr_log.see("end")))

    def _train_syn(self):
        self.tr_prog.start(); self.tr_log.insert("end","\n═══ Synthetic Training ═══\n")
        n=self.tr_n.get()
        threading.Thread(target=self._train_syn_w,args=(n,),daemon=True).start()

    def _train_syn_w(self,n):
        p,l=TrainingDataGenerator.generate(n)
        r=self.models.train(p,l,cb=self._tlog); self.models.save(); self.model_loaded=True
        self.root.after(0,lambda:(self.tr_prog.stop(),self._tlog(f"✓ Done! RF={r['rf']:.1%}"),
            self._refresh_dashboard(),self._update_status()))

    def _train_pcap(self):
        f=filedialog.askopenfilename(filetypes=[("PCAP","*.pcap *.pcapng *.cap")])
        if not f: return
        if not HAS_SCAPY: messagebox.showerror("Error","pip install scapy"); return
        self.tr_prog.start(); self.tr_log.insert("end",f"\n═══ PCAP Training: {Path(f).name} ═══\n")
        threading.Thread(target=self._train_pcap_w,args=(f,),daemon=True).start()

    def _train_pcap_w(self,path):
        try:
            fsize=os.path.getsize(path)
            self._tlog(f"File: {fsize/1024/1024:.1f} MB — streaming packets...")
            streams={}; stream_bytes={}; pkt_count=0; t0=time.time()
            try:
                reader=PcapReader(path)
            except:
                reader=rdpcap(path)
            for pkt in reader:
                pkt_count+=1
                if pkt_count%10000==0:
                    self._tlog(f"  {pkt_count:,} packets, {len(streams)} streams...")
                if not pkt.haslayer(IP): continue
                ip=pkt[IP]; sp=dp=0; pay=b''
                if pkt.haslayer(TCP): sp,dp=pkt[TCP].sport,pkt[TCP].dport
                elif pkt.haslayer(UDP): sp,dp=pkt[UDP].sport,pkt[UDP].dport
                if pkt.haslayer(Raw): pay=bytes(pkt[Raw].load)
                if not pay or len(pay)<16: continue
                k=(ip.src,ip.dst,sp,dp)
                cur=stream_bytes.get(k,0)
                if cur>=8192: continue  # Cap for training too
                if k not in streams: streams[k]=[]
                keep=pay[:8192-cur]
                streams[k].append(keep); stream_bytes[k]=cur+len(keep)
            if hasattr(reader,'close'): reader.close()
            self._tlog(f"Loaded {pkt_count:,} packets in {time.time()-t0:.1f}s — {len(streams)} streams")
            payloads=[]; labels=[]
            for k,plist in streams.items():
                combined=b''.join(plist)
                if len(combined)<64: continue
                f=FeatureExtractor.extract(combined)
                ent=f[0]; asc=f[23]; chi=f[2]; xor=f[34]
                if ent>7.5 and chi<310 and xor<2 and asc<0.5: lb=0
                elif ent>6.5 and chi<500: lb=1
                elif asc>0.8 or ent<4: lb=3
                else: lb=2
                payloads.append(combined[:8192]); labels.append(lb)
                ln=TrainingDataGenerator.LABELS[lb]; src,_,sp,dp=k
                self._tlog(f"  {src}:{sp}→{dp} → {ln} (ent={ent:.2f})")
                # Save as feedback
                ld=SAMPLES_DIR/{0:"strong",1:"moderate",2:"weak",3:"critical"}[lb]
                ld.mkdir(parents=True,exist_ok=True)
                (ld/f"pcap_{datetime.now().strftime('%H%M%S_%f')}.bin").write_bytes(combined[:8192])
            if not payloads: self._tlog("No usable streams"); self.root.after(0,self.tr_prog.stop); return
            syn_p,syn_l=TrainingDataGenerator.generate(max(100,max(collections.Counter(labels).values())))
            payloads.extend(syn_p); labels.extend(syn_l.tolist())
            self._tlog(f"Total: {len(payloads)} samples (real + synthetic)")
            r=self.models.train(payloads,np.array(labels),cb=self._tlog)
            self.models.save(); self.model_loaded=True
            self.root.after(0,lambda:(self.tr_prog.stop(),self._refresh_dashboard(),self._update_status()))
        except Exception as e:
            log.error(traceback.format_exc())
            self._tlog(f"Error: {e}"); self.root.after(0,self.tr_prog.stop)

    def _retrain_fb(self):
        n=sum(1 for _ in SAMPLES_DIR.rglob("*.bin"))
        if n<10: messagebox.showinfo("Info",f"Only {n} samples. Need 10+."); return
        self.tr_prog.start(); self.tr_log.insert("end",f"\n═══ Feedback Retrain ({n} samples) ═══\n")
        threading.Thread(target=self._retrain_fb_w,daemon=True).start()

    def _retrain_fb_w(self):
        lmap={"strong":0,"moderate":1,"weak":2,"critical":3}; P=[]; L=[]
        for name,lid in lmap.items():
            d=SAMPLES_DIR/name
            if not d.exists(): continue
            for f in d.glob("*.bin"):
                data=f.read_bytes()
                if len(data)>=32: P.append(data); L.append(lid)
        self._tlog(f"Loaded {len(P)} feedback samples")
        syn_p,syn_l=TrainingDataGenerator.generate(200); P.extend(syn_p); L.extend(syn_l.tolist())
        r=self.models.train(P,np.array(L),cb=self._tlog); self.models.save()
        self.root.after(0,lambda:(self.tr_prog.stop(),self._refresh_dashboard()))

    def _maybe_auto_train(self):
        """Auto-retrain if enough new samples have accumulated."""
        if not SAMPLES_DIR.exists(): return
        n=sum(1 for _ in SAMPLES_DIR.rglob("*.bin"))
        last=getattr(self,'_last_auto_train_count',0)
        if n >= last + 50 and n >= 20:  # 50 new samples since last train
            log.info(f"Auto-training: {n} samples ({n-last} new)")
            self._last_auto_train_count=n
            threading.Thread(target=self._auto_train_worker,daemon=True).start()

    def _auto_train_worker(self):
        """Background auto-retrain — doesn't touch UI."""
        try:
            lmap={"strong":0,"moderate":1,"weak":2,"critical":3}; P=[]; L=[]
            for name,lid in lmap.items():
                d=SAMPLES_DIR/name
                if not d.exists(): continue
                for f in list(d.glob("*.bin"))[:500]:  # Cap at 500 per class
                    data=f.read_bytes()
                    if len(data)>=32: P.append(data); L.append(lid)
            syn_p,syn_l=TrainingDataGenerator.generate(200); P.extend(syn_p); L.extend(syn_l.tolist())
            self.models.train(P,np.array(L))
            self.models.save()
            log.info(f"Auto-train complete: {len(P)} samples")
            self.root.after(0,self._refresh_dashboard)
        except Exception as e:
            log.error(f"Auto-train failed: {e}")

    # ── Settings ──
    def _tab_settings(self):
        t=ttk.Frame(self.nb); self.nb.add(t,text="  Settings  ")
        # Ollama
        of=tk.LabelFrame(t,text=" Ollama AI Validation ",bg=self.CARD_BG,fg=self.ACCENT,
            font=("Segoe UI",11,"bold"),bd=1,relief="solid")
        of.pack(fill="x",padx=16,pady=12)
        r1=tk.Frame(of,bg=self.CARD_BG); r1.pack(fill="x",padx=12,pady=8)
        tk.Label(r1,text="URL:",fg=self.FG,bg=self.CARD_BG).pack(side="left")
        self.s_url=tk.StringVar(value=self.settings["ollama_url"])
        ttk.Entry(r1,textvariable=self.s_url,width=35).pack(side="left",padx=4)
        ttk.Button(r1,text="Test",command=lambda:messagebox.showinfo("Ollama",
            "Connected ✓" if AIValidator(self.s_url.get(),self.s_model.get())._check() else "Cannot connect")).pack(side="left",padx=4)
        r2=tk.Frame(of,bg=self.CARD_BG); r2.pack(fill="x",padx=12,pady=(0,8))
        tk.Label(r2,text="Model:",fg=self.FG,bg=self.CARD_BG).pack(side="left")
        self.s_model=tk.StringVar(value=self.settings["ollama_model"])
        ttk.Entry(r2,textvariable=self.s_model,width=25).pack(side="left",padx=4)

        # Analysis
        af=tk.LabelFrame(t,text=" Analysis Options ",bg=self.CARD_BG,fg=self.ACCENT,
            font=("Segoe UI",11,"bold"),bd=1,relief="solid")
        af.pack(fill="x",padx=16,pady=(0,12))
        self.s_dec=tk.BooleanVar(value=self.settings.get("auto_decrypt",True))
        ttk.Checkbutton(af,text="Auto-decrypt weak streams",variable=self.s_dec).pack(anchor="w",padx=12,pady=4)
        self.s_val=tk.BooleanVar(value=self.settings.get("auto_validate",True))
        ttk.Checkbutton(af,text="Auto-validate with AI",variable=self.s_val).pack(anchor="w",padx=12,pady=4)

        # Paths
        pf=tk.LabelFrame(t,text=" Storage (all inside this folder) ",bg=self.CARD_BG,fg=self.ACCENT,
            font=("Segoe UI",11,"bold"),bd=1,relief="solid")
        pf.pack(fill="x",padx=16,pady=(0,12))
        for lbl,p in [("App",APP_DIR),("Models",MODELS_DIR),("Feedback",FEEDBACK_DIR),
                       ("Logs",LOGS_DIR),("Exports",EXPORTS_DIR),("Debug",DEBUG_DIR)]:
            tk.Label(pf,text=f"{lbl}: {p}",fg=self.FG2,bg=self.CARD_BG,font=("Consolas",8)).pack(anchor="w",padx=12,pady=1)

        br=tk.Frame(t,bg=self.BG); br.pack(fill="x",padx=16,pady=8)
        ttk.Button(br,text="💾 Save",style="Accent.TButton",command=self._apply_settings).pack(side="left")
        ttk.Button(br,text="📂 Open Folder",command=lambda:os.startfile(str(APP_DIR)) if sys.platform=="win32"
            else os.system(f'xdg-open "{APP_DIR}"')).pack(side="left",padx=8)
        ttk.Button(br,text="🗑 Clear Feedback",command=self._clear_fb).pack(side="right")
        ttk.Button(br,text="🔑 Clear Key Vault",command=self._clear_keys).pack(side="right",padx=4)
        ttk.Button(br,text="🧹 Clear Intel",command=self._clear_intel).pack(side="right",padx=4)

    def _apply_settings(self):
        self.settings.update(ollama_url=self.s_url.get(),ollama_model=self.s_model.get(),
            auto_decrypt=self.s_dec.get(),auto_validate=self.s_val.get())
        self.validator=AIValidator(self.settings["ollama_url"],self.settings["ollama_model"])
        self._save_settings(); self._update_status(); messagebox.showinfo("Saved","Settings saved")

    def _clear_fb(self):
        if messagebox.askyesno("Confirm","Clear all feedback?"):
            for d in SAMPLES_DIR.iterdir():
                if d.is_dir():
                    for f in d.glob("*"): f.unlink()
            self.feedback.stats={}
            ms=FEEDBACK_DIR/"method_stats.json"
            if ms.exists(): ms.unlink()
            self._refresh_dashboard()

    def _clear_keys(self):
        n=len(self.harvester.keys)
        if messagebox.askyesno("Clear Key Vault",f"Clear all {n} harvested keys?\n\nThis removes keys collected across all sessions.\nYou'll lose cross-capture reuse detection."):
            self.harvester.clear()
            self._refresh_dashboard()
            messagebox.showinfo("Key Vault",f"Cleared {n} keys")

    def _clear_intel(self):
        r=self.intel.get_report()
        n=r["total_credentials"]+r["total_cribs"]+r["total_patterns"]
        if messagebox.askyesno("Clear Intel Vault",f"Clear all collected intelligence?\n\n"
            f"  {r['total_credentials']} credentials\n  {r['total_cribs']} plaintext cribs\n  {r['total_patterns']} patterns\n\n"
            f"This removes auto-crib data used for future decryption."):
            self.intel.clear()
            self._refresh_dashboard()
            messagebox.showinfo("Intel Vault",f"Cleared {n} intel items")

    # ── Detail display ──
    def _show_detail(self,r,tw):
        tw.delete("1.0","end")
        tw.insert("end",f"{'═'*55}\n  {r.get('stream','Payload')}")
        if r.get("port"): tw.insert("end",f"  [{r['port']}]")
        tw.insert("end",f"\n  {r['bytes']:,} bytes | {r.get('packets','?')} packets")
        proto=r.get("protocol",{})
        if proto and proto.get("protocol"):
            tw.insert("end",f"\n  Protocol: {proto['protocol']} {proto.get('version','')}")
        tw.insert("end",f"\n  ML VERDICT: {r['label']} ({r['confidence']:.1%})")
        if r.get("protocol_override"):
            tw.insert("end",f"  [overridden — {r.get('protocol',{}).get('protocol','?')} detected]")
        if r.get("cracked_override"):
            tw.insert("end",f"  [upgraded — decryption successful]")
        tw.insert("end",f"\n{'═'*55}\n")

        # ══════════ FINDINGS — what was found (shown FIRST) ══════════
        proofs=[p for p in r.get("proofs",[]) if p.get("success")]
        pan=r.get("pan",[])

        if proofs or pan:
            tw.insert("end",f"\n  ★★★ FINDINGS ★★★\n  {'─'*50}\n")
            findings=[]; all_sensitive=[]; all_previews=[]

            # Card numbers from PAN scan
            for p in pan:
                findings.append(f"CARD NUMBER ({p['card'].upper()}): {p['masked']}")
                all_sensitive.append(("CARD NUMBER",p['masked']))

            for pr in proofs:
                m=pr.get("method","")
                # Collect sensitive items
                for lbl,val in pr.get("sensitive",[]):
                    all_sensitive.append((lbl.upper(),val))
                if pr.get("card_data"):
                    findings.append(f"Card data found inside {m} output")
                # Collect previews
                prev=pr.get("preview") or pr.get("final_preview","")
                if prev and pr.get("printable",0)>0.3:
                    all_previews.append((m,prev,pr.get("confidence",0)))
                # Human-readable finding description
                if m=="XOR Key Recovery":
                    findings.append(f"XOR ENCRYPTION BROKEN — key: {pr['key_hex'][:32]}{'...' if len(pr.get('key_hex',''))>32 else ''} ({pr.get('key_len','?')}B)")
                elif m=="Caesar/Shift":
                    findings.append(f"CAESAR CIPHER BROKEN — shift value: {pr.get('shift')}")
                elif m in ("ROT13","ROT47"):
                    findings.append(f"{m} DETECTED — trivial rotation cipher")
                elif m=="Vigenère Crack":
                    findings.append(f"VIGENÈRE BROKEN — keyword: \"{pr.get('key_letters','?')}\"")
                elif m=="Base64 Decode":
                    findings.append(f"NOT ENCRYPTED — Base64 encoded ({pr.get('decoded_size',0):,}B)")
                elif m=="Hex Decode":
                    findings.append(f"NOT ENCRYPTED — hex encoded ({pr.get('decoded_size',0):,}B)")
                elif m=="URL Decode":
                    findings.append("NOT ENCRYPTED — URL encoded")
                elif m=="Multi-Layer Decode":
                    layers=" → ".join(l["encoding"] for l in pr.get("layers",[]))
                    findings.append(f"ENCODING LAYERS STRIPPED: {layers}")
                elif m=="ECB Analysis":
                    findings.append(f"ECB MODE — {pr.get('unique','?')}/{pr.get('total','?')} unique blocks ({pr.get('ratio',0):.0%})")
                elif m=="Known-Plaintext Attack":
                    crib_note=" (auto-crib from intel vault)" if pr.get("auto_crib") else ""
                    findings.append(f"CRIB ATTACK — key: {pr.get('key_hex','')} (offset {pr.get('offset',0)}){crib_note}")
                elif m=="Cross-Stream Key Reuse":
                    src_cap=pr.get("source_capture","")
                    cap_note=f" [{src_cap}]" if src_cap else ""
                    findings.append(f"🔗 KEY REUSE — decrypted with key from: {pr.get('source_stream','?')}{cap_note}")
                elif m=="Known Pattern Match":
                    findings.append(f"📋 KNOWN PATTERN — seen {pr.get('times_seen',0)} times before, previously decrypts to: \"{pr.get('previously_decrypts_to','?')[:40]}\"")
                elif m=="DNS Decode":
                    qnames=pr.get("query_names",[])
                    ips=pr.get("resolved_ips",[])
                    qr=pr.get("qr","query")
                    rcode=pr.get("rcode","")
                    if qnames:
                        findings.append(f"🌐 DNS {qr.upper()} — {', '.join(qnames[:3])}")
                    if ips:
                        findings.append(f"   Resolved: {', '.join(ips[:5])}")
                    if rcode and rcode!="OK":
                        findings.append(f"   Status: {rcode}")
                elif m=="Pattern Detection":
                    pats=pr.get("repeating_patterns",[])
                    if pats: findings.append(f"REPEATING PATTERN — {pats[0]['length']}B × {pats[0]['ratio']:.0%}")
                elif m=="Entropy Windowing" and pr.get("mixed_content"):
                    findings.append(f"MIXED CONTENT — {pr.get('low_entropy_sections',0)} plaintext + {pr.get('high_entropy_sections',0)} encrypted sections")
                elif m=="Protocol Detection":
                    pv=pr.get("version","")
                    proto_name=pr.get("protocol","")
                    if proto_name=="STUN":
                        findings.append(f"📡 STUN/TURN — {pv} ({pr.get('msg_len',0)}B payload)")
                    elif pr.get("port_inferred"):
                        findings.append(f"✓ {proto_name} {pv} — inferred from port (high entropy)")
                    elif pr.get("weak_version"):
                        findings.append(f"⚠ {proto_name} {pv} — outdated version, consider upgrading")
                    else:
                        findings.append(f"✓ {proto_name} {pv} — properly encrypted protocol traffic")
                elif m=="String Extraction":
                    cov=pr.get("coverage",0); ns=pr.get("count",0)
                    if cov>0.5: findings.append(f"MOSTLY PLAINTEXT — {cov:.0%} readable ({ns} strings)")
                    elif pr.get("sensitive"): findings.append(f"SENSITIVE LEAK — {len(pr['sensitive'])} items in {ns} strings")
                elif m=="RC4 Bias Detection":
                    findings.append(f"⚠ RC4 STREAM CIPHER — χ²={pr.get('chi_squared',0):.0f}, zero bias={pr.get('zero_bias',0):.1%}")
                elif m=="Substitution Analysis":
                    findings.append(f"SUBSTITUTION CIPHER — frequency correlation {pr.get('frequency_correlation',0):.0%} with English")
                elif m=="AI Traffic Analysis":
                    findings.append(f"🤖 AI: {pr.get('protocol_guess','?')} — {pr.get('findings','')[:80]}")
                    if pr.get("suggested_approach"):
                        findings.append(f"   Suggested: {pr.get('suggested_approach','')[:80]}")
                elif m=="Plaintext Protocol":
                    proto_name=pr.get("protocol","?")
                    creds=pr.get("credentials",[])
                    if proto_name=="WeatherFlow":
                        findings.append(f"📡 WeatherFlow Tempest broadcast — {pr.get('msg_type','')} from {pr.get('device_serial','?')}")
                    elif proto_name=="SSDP":
                        findings.append(f"📡 SSDP/UPnP — {pr.get('server','device discovery')}")
                        if pr.get("location"): findings.append(f"  Location: {pr['location']}")
                    else:
                        findings.append(f"⚠ {proto_name} CLEARTEXT — no encryption at all!")
                    if creds:
                        findings.append(f"  🔑 {len(creds)} credential(s) exposed")
                    if pr.get("community_string"):
                        findings.append(f"  🔑 SNMP community: '{pr.get('community_string')}'")
                    if pr.get("sensitive_fields"):
                        findings.append(f"  🔑 {len(pr['sensitive_fields'])} sensitive field(s)")
                elif m=="TLS Handshake":
                    hs_type=pr.get("type","?")
                    if hs_type=="ClientHello":
                        sni=pr.get("sni","")
                        findings.append(f"🔍 TLS ClientHello → {sni or '(no SNI)'} | {pr.get('cipher_suites',0)} ciphers | JA3: {pr.get('ja3','')[:12]}...")
                        if pr.get("weak_ciphers",0) > 0:
                            findings.append(f"  ⚠ {pr['weak_ciphers']} weak cipher(s) offered")
                    elif hs_type=="ServerHello":
                        findings.append(f"🔍 TLS ServerHello → {pr.get('chosen_cipher','?')}")
                    elif hs_type=="Certificate":
                        subj=pr.get("subjects",[])
                        if subj: findings.append(f"🔍 TLS Cert: {', '.join(subj[:3])}")

            # Print each finding
            for finding in findings:
                tw.insert("end",f"\n  ▶ {finding}\n")

            # ── Sensitive data (prominent) ──
            if all_sensitive:
                seen=set(); uniq=[]
                for l,v in all_sensitive:
                    k=f"{l}:{v}"
                    if k not in seen: seen.add(k); uniq.append((l,v))
                tw.insert("end",f"\n  {'─'*50}\n  ⚠ SENSITIVE DATA ({len(uniq)} items)\n  {'─'*50}\n")
                for lbl,val in uniq:
                    tw.insert("end",f"    ✗ {lbl:15s} {val}\n")

            # ── Recovered plaintext (boxed, prominent) ──
            if all_previews:
                tw.insert("end",f"\n  {'─'*50}\n  RECOVERED PLAINTEXT\n  {'─'*50}\n")
                for method,preview,conf in all_previews:
                    tw.insert("end",f"\n  [{method}] confidence: {conf:.0%}\n")
                    tw.insert("end",f"  ┌{'─'*48}┐\n")
                    for ln in [preview[i:i+48] for i in range(0,min(len(preview),240),48)]:
                        padded=f"{ln:48s}" if len(ln)<=48 else ln[:48]
                        tw.insert("end",f"  │ {padded}│\n")
                    tw.insert("end",f"  └{'─'*48}┘\n")

            tw.insert("end",f"\n  {'═'*55}\n")
        elif r["label"]=="STRONG":
            tw.insert("end","\n  ✓ No weaknesses detected — appears properly encrypted.\n\n")

        # ══════════ PROOF DETAILS (technical) ══════════
        if proofs:
            tw.insert("end",f"\n  PROOF DETAILS ({len(proofs)} methods)\n  {'─'*50}\n")
            for pr in proofs:
                tw.insert("end",f"\n  🔓 {pr['method']}")
                if "confidence" in pr: tw.insert("end",f" — {pr['confidence']:.0%}")
                tw.insert("end","\n")
                if "key_hex" in pr:
                    tw.insert("end",f"    Key (hex):   {pr['key_hex'][:64]}\n")
                    if "key_ascii" in pr: tw.insert("end",f"    Key (ascii): {pr['key_ascii']}\n")
                    tw.insert("end",f"    Key length:  {pr.get('key_len','?')} bytes\n")
                if "key_letters" in pr:
                    tw.insert("end",f"    Keyword:     {pr['key_letters']} ({pr.get('key_len','?')} letters)\n")
                    if "word_hits" in pr: tw.insert("end",f"    Words found: {pr['word_hits']}\n")
                if "shift" in pr:
                    tw.insert("end",f"    Shift:       {pr['shift']} (0x{pr['shift']:02X})\n")
                    if "word_hits" in pr: tw.insert("end",f"    Words found: {pr['word_hits']}\n")
                if "offset" in pr: tw.insert("end",f"    Crib at:     offset {pr['offset']}\n")
                if "source_stream" in pr:
                    tw.insert("end",f"    Key source:  {pr['source_stream']}\n")
                    tw.insert("end",f"    Via method:  {pr.get('source_method','?')}\n")
                    tw.insert("end",f"    Src conf:    {pr.get('source_confidence',0):.0%}\n")
                if "candidates_found" in pr: tw.insert("end",f"    Candidates:  {pr['candidates_found']}\n")
                if "decoded_size" in pr: tw.insert("end",f"    Decoded:     {pr['decoded_size']:,} bytes\n")
                if "printable" in pr and pr["printable"]>0:
                    tw.insert("end",f"    Readable:    {'█'*int(pr['printable']*20)}{'░'*(20-int(pr['printable']*20))} {pr['printable']:.0%}\n")
                if "top_blocks" in pr:
                    tw.insert("end",f"    Blocks:      {pr['total']} total, {pr['unique']} unique\n")
                    for bl in pr["top_blocks"][:3]: tw.insert("end",f"      [{bl['count']}× {bl['pct']}%] {bl['hex']}  {bl['ascii']}\n")
                    if "map" in pr: tw.insert("end",f"    Map:         {pr['map'][:65]}\n")
                if "layers" in pr:
                    for layer in pr["layers"]: tw.insert("end",f"    Layer:       {layer['encoding']} {layer['input_size']}B→{layer['output_size']}B\n")
                if "repeating_patterns" in pr:
                    for rp in pr["repeating_patterns"][:3]:
                        tw.insert("end",f"    Pattern:     {rp['length']}B \"{rp['ascii']}\" ({rp['ratio']:.0%})\n")
                if "autocorrelation_peaks" in pr and pr["autocorrelation_peaks"]:
                    tw.insert("end","    Autocorr:    "+" ".join(f"lag{ac['lag']}={ac['correlation']:.3f}" for ac in pr["autocorrelation_peaks"][:3])+"\n")
                if "min_entropy" in pr:
                    tw.insert("end",f"    Entropy:     {pr['min_entropy']:.2f}–{pr['max_entropy']:.2f}\n")
                    if pr.get("plaintext_regions"):
                        for rg in pr["plaintext_regions"][:2]: tw.insert("end",f"      @{rg['offset']}: \"{rg['preview'][:50]}\"\n")
                if "samples" in pr:
                    tw.insert("end",f"    Strings:     {pr['count']} ({pr['coverage']:.0%})\n")
                    for s in pr["samples"][:3]: tw.insert("end",f"      \"{s[:65]}\"\n")
                if "domains" in pr:  # DNS Decode
                    tw.insert("end",f"    Type:        DNS {pr.get('qr','?')} ({pr.get('rcode','?')})\n")
                    tw.insert("end",f"    TX ID:       {pr.get('txid','?')}\n")
                    for d in pr.get("domains",[])[:5]:
                        tw.insert("end",f"    Query:       {d['name']} ({d['type']})\n")
                    for a in pr.get("answers",[])[:8]:
                        tw.insert("end",f"    Answer:      {a['name']} → {a['data']} ({a['type']}, TTL {a['ttl']})\n")
                if pr.get("method")=="RC4 Bias Detection":
                    tw.insert("end",f"    Chi-sq:      {pr.get('chi_squared',0):.1f} (uniform=256, RC4=280-400)\n")
                    tw.insert("end",f"    Zero bias:   {pr.get('zero_bias',0):.1%} (expected ~3-10% for RC4)\n")
                    tw.insert("end",f"    Entropy:     {pr.get('entropy',0):.3f}\n")
                if pr.get("method")=="Substitution Analysis":
                    tw.insert("end",f"    Freq corr:   {pr.get('frequency_correlation',0):.0%} match to English\n")
                    tw.insert("end",f"    Unique bytes: {pr.get('unique_bytes',0)}\n")
                    tw.insert("end",f"    Top byte:    {pr.get('top_byte','?')}\n")
                if pr.get("method")=="AI Traffic Analysis":
                    tw.insert("end",f"    Protocol:    {pr.get('protocol_guess','?')}\n")
                    tw.insert("end",f"    Encryption:  {pr.get('encryption_type','?')}\n")
                    tw.insert("end",f"    Risk:        {pr.get('risk_level','?')}\n")
                    tw.insert("end",f"    Findings:    {pr.get('findings','')[:120]}\n")
                    if pr.get("suggested_approach"):
                        tw.insert("end",f"    Approach:    {pr.get('suggested_approach','')[:120]}\n")
                if pr.get("method")=="Plaintext Protocol":
                    tw.insert("end",f"    Protocol:    {pr.get('protocol','?')}\n")
                    if pr.get("request_line"): tw.insert("end",f"    Request:     {pr['request_line'][:100]}\n")
                    for cred in pr.get("credentials",[])[:5]:
                        tw.insert("end",f"    CREDENTIAL:  {cred}\n")
                    for sf in pr.get("sensitive_fields",[])[:5:]:
                        tw.insert("end",f"    SENSITIVE:   {sf}\n")
                    if pr.get("community_string"):
                        tw.insert("end",f"    Community:   '{pr['community_string']}'\n")
                    if pr.get("client_id"):
                        tw.insert("end",f"    MQTT Client:  {pr['client_id']}\n")
                    if pr.get("function"):
                        tw.insert("end",f"    Function:    {pr['function']} (unit {pr.get('unit_id','')})\n")
                    if pr.get("device_serial"):
                        tw.insert("end",f"    Serial:      {pr['device_serial']}\n")
                    if pr.get("msg_type"):
                        tw.insert("end",f"    Msg Type:    {pr['msg_type']}\n")
                    if pr.get("hub_sn"):
                        tw.insert("end",f"    Hub:         {pr['hub_sn']}\n")
                    if pr.get("server"):
                        tw.insert("end",f"    Server:      {pr['server']}\n")
                    if pr.get("location"):
                        tw.insert("end",f"    Location:    {pr['location']}\n")
                if pr.get("method")=="TLS Handshake":
                    if pr.get("sni"): tw.insert("end",f"    SNI:         {pr['sni']}\n")
                    if pr.get("cipher_suites"): tw.insert("end",f"    Ciphers:     {pr['cipher_suites']} offered\n")
                    if pr.get("chosen_cipher"): tw.insert("end",f"    Selected:    {pr['chosen_cipher']}\n")
                    if pr.get("ja3"): tw.insert("end",f"    JA3:         {pr['ja3']}\n")
                    if pr.get("weak_ciphers",0): tw.insert("end",f"    ⚠ Weak:     {pr['weak_ciphers']} weak cipher(s)\n")
                    for s in pr.get("subjects",[])[:3]:
                        tw.insert("end",f"    Cert CN:     {s}\n")

        # ══════════ AI VALIDATION ══════════
        if r.get("validations"):
            tw.insert("end",f"\n  AI VALIDATION\n  {'─'*50}\n")
            for v in r["validations"]:
                bar="█"*int(v["score"]*30)+"░"*(30-int(v["score"]*30))
                tw.insert("end",f"  {bar} {v['score']:.0%} — {v['verdict']}\n")
                if v.get("data_types"): tw.insert("end",f"    Types: {', '.join(v['data_types'])}\n")
                if v.get("ai_reasoning"): tw.insert("end",f"    AI: {v['ai_reasoning']}\n")
                mc=v.get("ml_correction",{})
                if mc.get("should_correct"): tw.insert("end",f"    ⟳ {mc['reason']}\n")

        # ══════════ ML DETAILS (bottom) ══════════
        tw.insert("end",f"\n  ML DETAILS\n  {'─'*50}\n  Probabilities:\n")
        for c,p in r.get("probs",{}).items():
            tw.insert("end",f"    {c:10s} {'█'*int(p*30)}{'░'*(30-int(p*30))} {p:.1%}\n")
        tw.insert("end",f"  Autoencoder: {r.get('ae_score',0):.3f} {'⚠ ANOMALY' if r.get('ae_anomaly') else '✓ Normal'}\n")
        if r.get("triggers"):
            tw.insert("end","  Triggers:\n")
            for t in r["triggers"]: tw.insert("end",f"    ▸ {t}\n")
        tw.insert("end",f"\n  RAW FEATURES\n  {'─'*50}\n")
        for n in ["shannon_entropy","chi_square","serial_corr_1","block16_dup",
                   "xor_ioc_ratio","ascii_ratio","null_ratio","fft_peak","bigram_entropy"]:
            if n in r.get("features",{}): tw.insert("end",f"    {n:20s} {r['features'][n]}\n")

    # ── Export / Debug ──
    def _export_session(self):
        if not self._session_results:
            messagebox.showinfo("Info","No results to export yet"); return
        path=DebugExporter.export_session(self._session_results)
        messagebox.showinfo("Exported",f"Session saved to:\n{path}")

    def _export_debug(self):
        path=DebugExporter.export_debug_snapshot()
        messagebox.showinfo("Debug Export",f"Debug snapshot:\n{path}\n\nSend this file for troubleshooting.")

    def run(self): self.root.mainloop()

if __name__=="__main__":
    try:
        app=CryptoAuditGUI()
        app.run()
    except Exception as e:
        log.critical(f"Fatal: {traceback.format_exc()}")
        try: messagebox.showerror("Fatal Error",f"{e}\n\nCheck logs/ folder.")
        except: print(f"FATAL: {e}")
