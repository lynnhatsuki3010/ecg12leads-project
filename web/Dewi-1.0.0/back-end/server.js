// --- Import modules ---
const express = require("express");
const multer = require("multer");
const { exec } = require("child_process");
const path = require("path");
const fs = require("fs");
const cors = require("cors");

// --- Setup server ---
const app = express();
app.use(cors({ origin: "*" })); // Cho phép mọi nguồn
const upload = multer({ dest: "uploads/" });

app.get("/", (req, res) => {
    res.send("✅ ECG Detection Server is running...");
});

// --- Chỉ cho phép các file .mat hoặc .npy ---
const allowedExtensions = [".mat", ".npy"];

app.post("/detect", upload.single("file"), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: "⚠️ No file uploaded" });
    }

    const fileExt = path.extname(req.file.originalname).toLowerCase();
    if (!allowedExtensions.includes(fileExt)) {
        return res.status(400).json({ error: "❌ Unsupported file type. Please upload .mat or .npy" });
    }

    const filePath = req.file.path;

    console.log(`📂 Received file: ${req.file.originalname} (${fileExt})`);

    // --- Gọi Python detect ---
    exec(`python ./detect.py "${filePath}"`,{ encoding: "utf8" }, (error, stdout, stderr) => {
        if (error) {
            console.error("❌ Error running Python script:", stderr);
            return res.status(500).json({ error: "Lỗi khi chạy model" });
        }

        try {
            const result = JSON.parse(stdout.trim());
            console.log("✅ Detection result:", result);
            res.json(result);
        } catch (parseError) {
            console.error("❌ JSON parse error:", parseError);
            console.log("Raw output:", stdout);
            res.status(500).json({ error: "Lỗi phân tích kết quả từ model" });
        }

        // Xoá file tạm
        // fs.unlinkSync(filePath);
    });
});

app.listen(5000, () => {
    console.log("🚀 ECG Detection Server chạy tại http://localhost:5000");
});
