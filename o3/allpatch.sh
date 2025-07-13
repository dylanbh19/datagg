###############################################################################
# 1️⃣  Package skeleton  – safe on Git-Bash / PowerShell / WSL
###############################################################################
# explicit array avoids brace-expansion surprises on Windows shells
dirs=(
  "$PKG"
  "$PKG/data"
  "$PKG/processing"
  "$PKG/features"
  "$PKG/viz"
  "$PKG/utils"
  "$PKG/analytics"
  "$PKG/models"
)

echo "📂  Creating package directories …"
for d in "${dirs[@]}"; do
  mkdir -p "$d"
  # every package needs an __init__.py; create if absent
  [ -f "$d/__init__.py" ] || : > "$d/__init__.py"
done
echo "✅  Package skeleton ready."