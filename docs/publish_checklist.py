"""
# VirtualCell-Agent GitHub 发布清单

## 发布前检查

### 代码层面
- [ ] `python -m core.agent --query "NOTCH1 knockout"` 能正常运行
- [ ] `python -m core.agent --target SHH --perturbation drug` 输出合理报告
- [ ] `python tests/test_core.py` 全部通过

### 文档层面
- [ ] README.md 已更新为实际 GitHub URL
- [ ] LICENSE 文件存在
- [ ] requirements.txt 无缺失依赖
- [ ] `.gitignore` 文件存在

### 仓库层面
- [ ] GitHub repository 已创建
- [ ] 已添加 remote origin
- [ ] 已推送至 main 分支
- [ ] GitHub Pages / Actions 已配置（可选）

## .gitignore 建议内容
```
__pycache__/
*.pyc
.env
*.egg-info/
dist/
build/
output/reports/
.vscode/
.idea/
*.h5ad
*.sbml
*.npy
```
"""

CHECKLIST_MARKDOWN = """
# ✅ VirtualCell-Agent GitHub 发布清单

## 发布前检查

- [ ] `python -m core.agent --query "NOTCH1 knockout"` 运行正常
- [ ] `python tests/test_core.py` 全部通过
- [ ] README.md 中的 GitHub URL 已更新
- [ ] `.gitignore` 已创建
- [ ] 已验证知识库内容完整
"""
