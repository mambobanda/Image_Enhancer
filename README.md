Got you, Mambo — here is **ONE single Markdown file**, fully combined, fully formatted, and ready for copy‑paste as your `README.md`.  
Everything from both tools is included: enhancement pipeline + sharpening suite + usage + presets + explanations.

No extra fluff. No broken formatting.  
Just a clean, professional Markdown document.

---

# 📘 Image Enhancement & Sharpening Tools  
### *Complete Usage Guide (Batch Processing)*

This repository contains two complementary tools:

1. **Enhancement Pipeline Tool**  
   Full restoration workflow: *Deblur → Denoise → Upscale → Sharpen*

2. **Dedicated Sharpening Tool**  
   Multiple sharpening modes inspired by *DaVinci Resolve FX Sharpen*

Both tools support **batch folder processing**, preserve directory structure, and provide fine‑grained control over image detail.

---

---

## 🟦 1. Enhancement Pipeline Tool

### **Command**
```bash
python enhance.py <input_folder> <output_folder> [options]
```

This tool performs:

- Deblur (NAFNet GoPro)
- Denoise (SIDD)
- Upscale (EDSR)
- Sharpen (Resolve‑style)
- GPU acceleration (optional)
- Mixed precision (VRAM‑optimized)
- Automatic memory cleanup

---

### 🔧 Options

#### **Deblur / Denoise**
| Option | Description |
|--------|-------------|
| `--no-deblur` | Disable deblurring |
| `--no-denoise` | Disable denoising |
| `--deblur-factor <float>` | Deblur strength *(recommended: 0.3–0.6)* |
| `--denoise-factor <float>` | Denoise strength *(recommended: 0.3–0.5)* |

#### **Upscaling**
| Option | Description |
|--------|-------------|
| `--no-scale` | Disable upscaling |
| `--scale <int>` | Upscale factor *(2, 3, 4)* |

#### **Sharpening**
| Option | Description |
|--------|-------------|
| `--sharpen` | Enable sharpening |
| `--sharpen-amount <float>` | Sharpen intensity |
| `--sharpen-radius <float>` | Sharpen radius |

---

### ⭐ Recommended Settings  
*(No neon edges, no segmentation)*

#### **Natural detail (best overall)**
```bash
--deblur-factor 0.4
--denoise-factor 0.3
--sharpen --sharpen-amount 0.6 --sharpen-radius 1.4
```

#### **Soft sharpening (portraits)**
```bash
--sharpen --sharpen-amount 0.4 --sharpen-radius 1.6
```

#### **Crisp detail (architecture, product shots)**
```bash
--sharpen --sharpen-amount 1.0 --sharpen-radius 0.8
```

#### **Maximum natural look (no artifacts)**
```bash
--no-deblur
--denoise-factor 0.3
--sharpen --sharpen-amount 0.7 --sharpen-radius 1.4
```

---

---

## 🟩 2. Dedicated Sharpening Tool

This tool applies sharpening **only**, without deblur, denoise, or upscale.  
It includes four sharpening modes inspired by *DaVinci Resolve FX*.

### **Command**
```bash
python sharpen_tool.py <input_folder> <output_folder> --mode <type> [options]
```

---

## 🔧 Sharpening Modes

### 1️⃣ Unsharp Mask  
Classic Resolve-style sharpening.

```
--mode unsharp
--amount <float>
--radius <float>
--threshold <int>
```

**Best for:** general sharpening, crisp edges  
**Example:**
```bash
python sharpen_tool.py in out --mode unsharp --amount 1.0 --radius 1.2
```

---

### 2️⃣ Frequency Sharpen  
Multi-frequency sharpening (fine, medium, large detail).

```
--mode frequency
--amount <float>
```

**Best for:** natural detail, no halos  
**Example:**
```bash
python sharpen_tool.py in out --mode frequency --amount 0.8
```

---

### 3️⃣ Edge-Aware Sharpen  
Sharpens only detected edges.

```
--mode edges
--amount <float>
--radius <float>
--edge-threshold <int>
```

**Best for:** avoiding neon edges  
**Example:**
```bash
python sharpen_tool.py in out --mode edges --amount 1.0 --radius 1.0
```

---

### 4️⃣ Selective Sharpen  
Sharpen only shadows, highlights, or midtones.

```
--mode selective
--amount <float>
--radius <float>
--sharpen-highlights
--sharpen-shadows
--threshold-low <int>
--threshold-high <int>
```

**Best for:** portraits, cinematic looks  
**Example:**
```bash
python sharpen_tool.py in out --mode selective --amount 1.0 --radius 1.2 --sharpen-highlights
```

---

---

## ⭐ Sharpening Presets

### ✔ Natural (no halos)
```bash
--mode frequency --amount 0.8
```

### ✔ Crisp detail
```bash
--mode unsharp --amount 1.0 --radius 1.0
```

### ✔ Anti-neon edges
```bash
--mode edges --amount 0.6 --radius 1.4
```

### ✔ Portrait-safe
```bash
--mode selective --amount 0.5 --radius 1.6 --sharpen-highlights
```

---

---

## 🧠 Notes

- Both tools support **full folder batch processing**  
- Both tools preserve directory structure  
- Sharpening-only tool is CPU-only and lightweight  
- Enhancement tool uses GPU (optional) and mixed precision  
- Recommended presets avoid neon edges and segmentation  
- Sharpening modes mimic DaVinci Resolve behavior  
- Tools are safe for large batches (VRAM-optimized)

---
