# CryptoAudit ML

**AI-powered encryption analyzer and network traffic decryption toolkit.**

CryptoAudit ML analyzes network traffic (PCAP files or live capture) to identify weak, broken, or missing encryption — then attempts to crack it. It combines a 5-model ML ensemble for traffic classification with 15+ decryption methods, persistent key/credential harvesting, and optional Ollama AI integration for traffic analysis.

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

## What It Does

- **Classifies encrypted traffic** — ML ensemble distinguishes proper TLS/SSH from weak XOR, Caesar, substitution ciphers, and plaintext protocols
- **Cracks weak encryption** — 15+ decryption methods including XOR key recovery, Caesar/shift, Vigenère, Base64/Hex decode, ECB block analysis, RC4 bias detection, byte substitution analysis, and known-plaintext attacks
- **Detects cleartext protocols** — HTTP, FTP, Telnet, SMTP, SNMP (community strings), MQTT, Modbus TCP, SIP/RTSP, SSDP/UPnP, WeatherFlow, and DNS
- **Parses TLS handshakes** — Extracts SNI hostnames, cipher suites, JA3 fingerprints, and certificate subjects without decryption
- **Harvests credentials** — Extracts usernames, passwords, HTTP Basic Auth, SMTP AUTH, FTP credentials, PAN/credit card numbers (Luhn-validated), and API tokens
- **Cross-stream key reuse detection** — Recovered keys are automatically tested against all other streams in iterative passes
- **Persistent intelligence** — Keys, credentials, and plaintext cribs persist across sessions for cross-capture analysis
- **Optional AI analysis** — Ollama integration for AI-powered traffic identification and decryption suggestions

## Quick Start

### Windows
```
Double-click CryptoAudit.bat
```
The launcher checks for Python, installs missing dependencies automatically, and starts the GUI.

### Linux / macOS
```bash
pip install numpy scikit-learn scipy joblib
pip install scapy  # Optional, for live capture
python3 crypto_audit.py
```

### Live Capture (requires admin/root)
```bash
# Windows: Run as Administrator
# Linux: sudo python3 crypto_audit.py
```

## Requirements

- **Python 3.10+**
- **numpy**, **scikit-learn**, **scipy**, **joblib** (installed automatically by launcher)
- **scapy** (optional, for live packet capture)
- **Ollama** (optional, for AI traffic analysis — runs locally)

## Features

### ML Classification Pipeline
5-model ensemble (Random Forest, Gradient Boosting, Autoencoder, Isolation Forest) with 37 extracted features. Auto-trains on first launch with synthetic data, then retrains from real network samples as you analyze captures.

### Decryption Methods
| Method | What It Finds |
|--------|--------------|
| XOR Key Recovery | Single and multi-byte XOR with frequency analysis |
| Caesar/Shift | Byte-level shift ciphers (ROT13, ROT47, arbitrary) |
| Vigenère Crack | Polyalphabetic substitution via Kasiski + chi-squared |
| Base64 / Hex / URL Decode | Encoded (not encrypted) data |
| Multi-Layer Decode | Nested encodings (Base64 wrapping Hex, etc.) |
| ECB Block Analysis | Repeated ciphertext blocks indicating ECB mode |
| Known-Plaintext Attack | Crib-based key recovery with auto-cribs |
| RC4 Bias Detection | Statistical detection of RC4 stream cipher output |
| Substitution Analysis | Frequency correlation against English distribution |
| Pattern Detection | Repeating byte patterns indicating weak keys |
| Entropy Windowing | Mixed plaintext/ciphertext regions |
| String Extraction | Readable strings leaked in encrypted streams |

### Protocol Detection
| Protocol | Detection Method |
|----------|-----------------|
| TLS/SSL | Header parsing, mid-stream markers, port inference |
| SSH | Banner detection, port-based inference |
| STUN/TURN | Magic cookie 0x2112A442, message type parsing |
| WireGuard | Message type identification |
| DNS | Full packet decode with pointer compression |
| HTTP | Header parsing, Basic Auth extraction, POST body scanning |
| FTP | USER/PASS credential extraction |
| Telnet | Command/keystroke capture |
| SMTP | AUTH PLAIN decoding, email extraction |
| SNMP | Community string extraction (v1/v2c) |
| MQTT | Client ID, topic, and payload parsing |
| Modbus TCP | Function code decoding, zero-encryption flagging |
| SIP/RTSP | URI and Digest auth extraction |
| SSDP/UPnP | Device discovery, location/server parsing |
| WeatherFlow | Tempest station JSON broadcast parsing |

### TLS Handshake Intelligence
Extracts from unencrypted handshake data (no decryption needed):
- **SNI hostnames** — what domain the client requested
- **Cipher suites** — weak cipher detection (RC4, DES, export)
- **JA3 fingerprints** — client application identification
- **Certificate subjects** — server identity from Certificate messages
- **Forward secrecy** — flags RSA key exchange (no PFS)

### Key & Intel Persistence
- **Key Vault** — harvested keys persist in `feedback/key_vault.json` across sessions
- **Intel Vault** — credentials, plaintext cribs, and pattern fingerprints in `feedback/intel_vault.json`
- **Cross-capture reuse** — detect the same XOR key used across different captures
- **Prefix clustering** — identify key families sharing common prefixes
- **Auto-cribs** — successful decryptions become cribs for attacking other streams

### Performance
- Per-stream memory cap (32KB) for long captures
- STRONG stream data eviction after analysis
- TreeView limiting for responsive GUI during live capture
- Parallel AI analysis (4 threads, Phase 4)
- Background auto-retraining every 50 new samples

## GUI Tabs

| Tab | Purpose |
|-----|---------|
| **Dashboard** | Live stats, key vault viewer, intel vault viewer |
| **PCAP Analysis** | Load and analyze capture files (4-phase pipeline) |
| **Live Capture** | Real-time packet capture and analysis |
| **Payload Analysis** | Paste hex/base64/text for manual analysis |
| **Training** | Synthetic training, PCAP training, feedback retraining |
| **Settings** | Ollama config, auto-decrypt, auto-validate toggles |

## Analysis Pipeline (PCAP)

1. **Phase 1 — Load**: Parse packets, reassemble streams, cap at 32KB per stream
2. **Phase 2 — Analyze**: Protocol detection → plaintext detection → ML classification → decryption → validation → key/intel harvesting
3. **Phase 3 — Cross-Stream**: Iterative key reuse attack (recovered keys tested against all uncracked non-TLS streams, loops until no new discoveries)
4. **Phase 4 — AI Analysis**: Parallel Ollama analysis of interesting unidentified streams (optional)

## Ollama Integration

If you have [Ollama](https://ollama.ai) running locally, CryptoAudit can use it for:
- **Traffic analysis** — identify unknown protocols and suggest decryption approaches
- **Decryption validation** — confirm whether decrypted output is real plaintext

Configure in the Settings tab. Default: `http://localhost:11434` with any model (e.g., `qwen2.5:14b`, `llama3`, `mistral`).

## File Structure

```
CryptoAudit/
├── crypto_audit.py      # Main application
├── CryptoAudit.bat      # Windows launcher
├── README.md
├── LICENSE
├── .gitignore
├── models/              # Trained ML models (auto-generated)
├── feedback/
│   ├── key_vault.json   # Persistent harvested keys
│   ├── intel_vault.json # Persistent credentials/cribs
│   ├── method_stats.json# Decryption method statistics
│   └── samples/         # Training samples by classification
│       ├── strong/
│       ├── moderate/
│       ├── weak/
│       └── critical/
├── exports/             # Session exports (JSON)
├── debug/               # Debug archives
└── logs/                # Application logs
```

## Use Cases

- **Security assessments** — Identify weak encryption on enterprise networks
- **IoT security auditing** — Find devices using plaintext or trivial encryption
- **Network forensics** — Analyze PCAP captures for credential exposure
- **BAS/ICS security** — Detect unencrypted BACnet, Modbus, MQTT traffic
- **Encryption compliance** — Verify TLS versions, cipher suites, and certificate validity

## Contributing

Pull requests welcome. Please include test cases for new decryption methods or protocol detectors.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Disclaimer

This tool is intended for authorized security testing and network analysis on networks you own or have explicit permission to test. Unauthorized interception of network traffic may violate federal and state wiretap laws. Always obtain proper authorization before analyzing network traffic.
