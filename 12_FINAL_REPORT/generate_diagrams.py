import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import os

os.makedirs('12_FINAL_REPORT/diagrams', exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

def draw_box(ax, x, y, w, h, text, color='#1f77b4', text_color='white'):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor='black', lw=1.5))
    ax.text(x + w/2, y + h/2, text, color=text_color, ha='center', va='center', weight='bold', fontsize=10)

def draw_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=15, color='black', lw=1.5)
    ax.add_patch(arrow)

# Draw Architecture Diagram
draw_box(ax, 4, 9, 2, 0.8, 'Camera / Video Input', color='#ff7f0e')

draw_arrow(ax, 5, 9, 5, 8.5)
draw_box(ax, 4, 7.7, 2, 0.8, 'Person Detection')

draw_arrow(ax, 5, 7.7, 5, 7.2)
draw_box(ax, 4, 6.4, 2, 0.8, 'ByteTrack (Tracking)')

draw_arrow(ax, 5, 6.4, 5, 5.9)
draw_box(ax, 4, 5.1, 2, 0.8, 'PPE Detection')

draw_arrow(ax, 5, 5.1, 5, 4.6)
draw_box(ax, 4, 3.8, 2, 0.8, 'Person-PPE Association')

draw_arrow(ax, 5, 3.8, 5, 3.3)
draw_box(ax, 4, 2.5, 2, 0.8, 'Zone / Rule Engine')

draw_arrow(ax, 5, 2.5, 5, 2.0)
draw_box(ax, 4, 1.2, 2, 0.8, 'Temporal Validation')

draw_arrow(ax, 5, 1.2, 5, 0.7)
draw_box(ax, 4, -0.1, 2, 0.8, 'Violation Decision', color='#d62728')

# Branch to right side
draw_arrow(ax, 6, 0.3, 7.5, 0.3)
draw_box(ax, 7.5, -0.1, 2, 0.8, 'Evidence Capture', color='#2ca02c')

draw_arrow(ax, 8.5, 0.7, 8.5, 1.2)
draw_box(ax, 7.5, 1.2, 2, 0.8, 'PostgreSQL', color='#9467bd')

draw_arrow(ax, 8.5, 2.0, 8.5, 2.5)
draw_box(ax, 7.5, 2.5, 2, 0.8, 'FastAPI', color='#8c564b')

draw_arrow(ax, 8.5, 3.3, 8.5, 3.8)
draw_box(ax, 7.5, 3.8, 2, 0.8, 'Next.js Dashboard', color='#17becf')

plt.title('EdgeVision System Architecture', fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig('12_FINAL_REPORT/diagrams/architecture.png', dpi=300, bbox_inches='tight')
plt.close()

# Draw Deployment Path
fig, ax = plt.subplots(figsize=(8, 3))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis('off')

draw_box(ax, 0.5, 1, 1.5, 1, 'PyTorch\n(.pt)', color='#1f77b4')
draw_arrow(ax, 2.0, 1.5, 2.5, 1.5)
draw_box(ax, 2.5, 1, 1.5, 1, 'ONNX\n(READY)', color='#ff7f0e')
draw_arrow(ax, 4.0, 1.5, 4.5, 1.5)
draw_box(ax, 4.5, 1, 2.0, 1, 'TensorRT FP16\n(INSTRUCTIONS READY)', color='#2ca02c')
draw_arrow(ax, 6.5, 1.5, 7.0, 1.5)
draw_box(ax, 7.0, 1, 2.5, 1, 'Jetson Orin\n(PHYSICAL VAL PENDING)', color='#d62728')

plt.title('Deployment Pipeline', fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig('12_FINAL_REPORT/diagrams/deployment.png', dpi=300, bbox_inches='tight')
plt.close()
