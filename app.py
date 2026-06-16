from flask import Flask, render_template, request, jsonify, send_file
import pickle, numpy as np, pandas as pd, json, io, os, shap
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from datetime import datetime

app   = Flask(__name__)
BASE  = os.path.dirname(__file__)

# ── Load artifacts ────────────────────────────────────────────────────────────
model     = pickle.load(open(f'{BASE}/models/xgb_tuned.pkl',    'rb'))
scaler    = pickle.load(open(f'{BASE}/models/scaler_v2.pkl',    'rb'))
explainer = pickle.load(open(f'{BASE}/models/shap_explainer.pkl','rb'))
le_map    = pickle.load(open(f'{BASE}/models/le_map.pkl',       'rb'))
meta      = json.load(open(f'{BASE}/models/results_v2.json'))
FEATURES  = meta['feature_names']

# ── Encoding maps ─────────────────────────────────────────────────────────────
GRADE_MAP  = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6}
EMP_MAP    = {'< 1 year':0,'1 year':1,'2 years':2,'3 years':3,'4 years':4,
              '5 years':5,'6 years':6,'7 years':7,'8 years':8,'9 years':9,'10+ years':10}
HOME_MAP   = {'MORTGAGE':0,'OTHER':1,'OWN':2,'RENT':3}
VERIF_MAP  = {'Not Verified':0,'Source Verified':1,'Verified':2}
PURPOSE_MAP= {'car':0,'credit_card':1,'debt_consolidation':2,'educational':3,
              'home_improvement':4,'house':5,'major_purchase':6,'medical':7,
              'moving':8,'other':9,'small_business':10,'vacation':11}

def encode_row(d):
    return np.array([[
        float(d['loan_amnt']),      float(d['term']),
        float(d['int_rate']),       float(d['installment']),
        GRADE_MAP[d['grade']],      EMP_MAP[d['emp_length']],
        HOME_MAP[d['home_ownership']], float(d['annual_inc']),
        VERIF_MAP[d['verification_status']], PURPOSE_MAP[d['purpose']],
        float(d['dti']),            float(d['delinq_2yrs']),
        float(d['inq_last_6mths']), float(d['open_acc']),
        float(d['pub_rec']),        float(d['revol_bal']),
        float(d['revol_util']),     float(d['total_acc']),
        float(d['fico_range_low']), float(d['fico_range_high'])
    ]])

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', meta=meta)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Single prediction + SHAP explanation."""
    d    = request.json
    feat = encode_row(d)
    prob = float(model.predict_proba(feat)[0][1])
    label= 'HIGH RISK' if prob > 0.5 else 'LOW RISK'

    # SHAP for this row
    sv   = explainer.shap_values(pd.DataFrame(feat, columns=FEATURES))
    shap_dict = {f: round(float(v),4) for f,v in zip(FEATURES, sv[0])}
    top_pos = sorted(shap_dict.items(), key=lambda x:-x[1])[:5]
    top_neg = sorted(shap_dict.items(), key=lambda x:x[1])[:5]

    return jsonify({
        'probability': round(prob*100,1),
        'label': label,
        'shap_top_positive': top_pos,
        'shap_top_negative': top_neg,
        'shap_all': shap_dict
    })

@app.route('/api/generate_letter', methods=['POST'])
def generate_letter():
    """Generate PDF approval / rejection letter."""
    d    = request.json
    feat = encode_row(d)
    prob = float(model.predict_proba(feat)[0][1])
    approved = prob <= 0.5

    buf = io.BytesIO()
    c   = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    # Header bar
    c.setFillColor(colors.HexColor('#0D1117'))
    c.rect(0, H-3*cm, W, 3*cm, fill=1, stroke=0)
    c.setFillColor(colors.HexColor('#58A6FF'))
    c.setFont('Helvetica-Bold', 22)
    c.drawString(2*cm, H-1.9*cm, 'LoanSight AI')
    c.setFont('Helvetica', 11)
    c.setFillColor(colors.HexColor('#8B949E'))
    c.drawString(2*cm, H-2.6*cm, 'Intelligent Credit Risk Assessment')

    # Date & ref
    c.setFillColor(colors.black)
    c.setFont('Helvetica', 10)
    ref = f"REF-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    c.drawRightString(W-2*cm, H-3.8*cm, f"Date: {datetime.now().strftime('%B %d, %Y')}")
    c.drawRightString(W-2*cm, H-4.5*cm, f"Reference: {ref}")

    # Decision banner
    dec_color = colors.HexColor('#0d2318') if approved else colors.HexColor('#2a0e0e')
    bdr_color = colors.HexColor('#3FB950') if approved else colors.HexColor('#F85149')
    c.setFillColor(dec_color)
    c.roundRect(2*cm, H-7.5*cm, W-4*cm, 2.2*cm, 8, fill=1, stroke=0)
    c.setStrokeColor(bdr_color); c.setLineWidth(1.5)
    c.roundRect(2*cm, H-7.5*cm, W-4*cm, 2.2*cm, 8, fill=0, stroke=1)
    c.setFillColor(bdr_color)
    c.setFont('Helvetica-Bold', 18)
    decision_text = '✓  LOAN APPROVED' if approved else '✗  LOAN DECLINED'
    c.drawCentredString(W/2, H-6.2*cm, decision_text)
    c.setFont('Helvetica', 11)
    c.setFillColor(colors.HexColor('#8B949E'))
    c.drawCentredString(W/2, H-6.9*cm, f"Default Risk Score: {prob*100:.1f}%  |  Threshold: 50.0%")

    # Body
    c.setFillColor(colors.black)
    c.setFont('Helvetica', 11)
    body_y = H-9*cm
    salutation = "Dear Applicant,"
    c.drawString(2*cm, body_y, salutation)
    body_y -= 0.7*cm

    if approved:
        body = (f"We are pleased to inform you that your loan application for "
                f"${float(d['loan_amnt']):,.0f} has been APPROVED following our "
                f"AI-powered credit assessment. Your profile demonstrates a low "
                f"default probability of {prob*100:.1f}%, which meets our acceptance criteria.")
    else:
        body = (f"After careful review of your application for ${float(d['loan_amnt']):,.0f}, "
                f"we regret to inform you that it has been DECLINED at this time. "
                f"Our model assessed a default probability of {prob*100:.1f}%, which "
                f"exceeds our acceptable threshold. We encourage you to reapply after "
                f"improving your credit profile.")

    # Word wrap
    words = body.split(); line = ''; lines = []
    for w in words:
        test = line+' '+w if line else w
        if c.stringWidth(test,'Helvetica',11) < W-4*cm: line=test
        else: lines.append(line); line=w
    if line: lines.append(line)
    for ln in lines:
        c.drawString(2*cm, body_y, ln); body_y -= 0.55*cm

    # Loan details table
    body_y -= 0.5*cm
    c.setFont('Helvetica-Bold', 12)
    c.setFillColor(colors.HexColor('#58A6FF'))
    c.drawString(2*cm, body_y, 'Application Details')
    body_y -= 0.4*cm
    c.setStrokeColor(colors.HexColor('#30363D')); c.setLineWidth(0.5)
    c.line(2*cm, body_y, W-2*cm, body_y); body_y -= 0.5*cm

    rows = [
        ('Loan Amount',     f"${float(d['loan_amnt']):,.0f}"),
        ('Term',            f"{d['term']} months"),
        ('Interest Rate',   f"{d['int_rate']}%"),
        ('Annual Income',   f"${float(d['annual_inc']):,.0f}"),
        ('Grade',           d['grade']),
        ('DTI Ratio',       f"{d['dti']}%"),
        ('FICO Score',      str(d['fico_range_low'])),
        ('Purpose',         d['purpose'].replace('_',' ').title()),
    ]
    c.setFont('Helvetica', 10)
    for i,(k,v) in enumerate(rows):
        bg = colors.HexColor('#161B22') if i%2==0 else colors.HexColor('#0D1117')
        c.setFillColor(bg)
        c.rect(2*cm, body_y-0.45*cm, W-4*cm, 0.5*cm, fill=1, stroke=0)
        c.setFillColor(colors.HexColor('#8B949E'))
        c.drawString(2.3*cm, body_y-0.3*cm, k)
        c.setFillColor(colors.HexColor('#E6EDF3'))
        c.drawRightString(W-2.3*cm, body_y-0.3*cm, v)
        body_y -= 0.5*cm

    # Footer
    c.setFillColor(colors.HexColor('#0D1117'))
    c.rect(0, 0, W, 2*cm, fill=1, stroke=0)
    c.setFillColor(colors.HexColor('#8B949E'))
    c.setFont('Helvetica', 8)
    c.drawCentredString(W/2, 1.2*cm, 'LoanSight AI  |  AI-Powered Credit Assessment  |  This is a portfolio demonstration.')
    c.drawCentredString(W/2, 0.7*cm, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  |  {ref}')

    c.save()
    buf.seek(0)
    fname = f"loan_{'approval' if approved else 'rejection'}_{ref[-8:]}.pdf"
    return send_file(buf, mimetype='application/pdf',
                     as_attachment=True, download_name=fname)

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction from uploaded CSV."""
    f   = request.files['file']
    df  = pd.read_csv(f)
    out = []
    for _, row in df.iterrows():
        try:
            d    = row.to_dict()
            feat = encode_row(d)
            prob = float(model.predict_proba(feat)[0][1])
            out.append({**d, 'default_probability': round(prob*100,1),
                        'risk_label': 'HIGH RISK' if prob>0.5 else 'LOW RISK'})
        except Exception as e:
            out.append({**row.to_dict(), 'default_probability':'ERROR','risk_label':str(e)})

    result_df = pd.DataFrame(out)
    buf = io.BytesIO()
    result_df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype='text/csv',
                     as_attachment=True, download_name='batch_predictions.csv')

@app.route('/api/whatif', methods=['POST'])
def whatif():
    """What-if: return probability for modified features."""
    d    = request.json
    feat = encode_row(d)
    prob = float(model.predict_proba(feat)[0][1])
    return jsonify({'probability': round(prob*100, 1),
                    'label': 'HIGH RISK' if prob > 0.5 else 'LOW RISK'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
