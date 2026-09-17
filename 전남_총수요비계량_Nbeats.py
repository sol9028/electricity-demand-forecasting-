"""
전남 총수요·비계량 예측 (형 방법론 · N-BEATS · 시간단위)
================================================================
[분해]  비계량 = K_r × s,  s = 계량태양광 ÷ 계량설비용량
        총수요_0 = 시장수요 + 비계량   (시장수요 = 순부하 + 계량)
[예측]  총수요 → N-BEATS (기온·달력·과거값만 / 태양광·일사·운량 제외)
        비계량 → K_r × s (계량 프로파일이 모양 결정)

분할: 학습 2023~2024 / 검증 2025
"""
import pandas as pd, numpy as np, holidays, warnings
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore'); torch.manual_seed(42); np.random.seed(42)

REGION_KO, REGION_GEN = '전남', '전라남도'
MIN5=['0분','5분','10분','15분','20분','25분','30분','35분','40분','45분','50분','55분']
LOOKBACK, HORIZON, EPOCHS = 168, 24, 18

# ═══ 1. 로딩 ═══
def load5(path, region, col):
    x=pd.read_excel(path); x=x[x['지역']==region].copy()
    x['dt']=pd.to_datetime(x['거래일'])+pd.to_timedelta(x['시간'],unit='h')
    x[col]=x[MIN5].mean(axis=1)/1e6*12   # Wh(5분)→순간MW
    return x[['dt',col]]

solar=load5('KPX_2020_2025년도_태양광_5분단위_발전량.xlsx',REGION_GEN,'solar')
wind =load5('KPX_2020_2025_풍력_지역별_5분단위__발전량.xlsx',REGION_GEN,'wind')

cap=pd.read_excel('설비용량.xlsx',header=None,skiprows=2)
cap.columns=['기간','지역','태양_계량','PPA','자가용']
cap['기간']=pd.to_datetime(cap['기간'])
cap=cap[cap['지역']==REGION_KO].dropna(subset=['태양_계량']).copy()
cap['ym']=cap['기간'].dt.to_period('M')
cap_m=cap.set_index('ym')[['태양_계량','PPA','자가용']]

load=pd.read_csv('KPX_2020_2025년도_시간별_전국_전력수요량.csv',encoding='utf-8-sig')
ll=load.melt(id_vars=['날짜'],var_name='h',value_name='nat')
ll['h']=ll['h'].str.replace('시','',regex=False).astype(int)
ll['nat']=pd.to_numeric(ll['nat'].astype(str).str.replace(',','',regex=False),errors='coerce')
ll['dt']=pd.to_datetime(ll['날짜'])+pd.to_timedelta(ll['h']-1,unit='h')
ll=ll[['dt','nat']].dropna()

ratio=pd.read_csv('region_monthly_ratio_2023_2025.csv',encoding='utf-8-sig')
jr=ratio[ratio['시도']==REGION_GEN][['year','month','ratio']]

# 기상 (광주 156, 태양광무관: 기온·습도·풍속만)
wthr=pd.read_csv('전남지역_기상데이터.csv',encoding='cp949',low_memory=False)
wthr=wthr[wthr['지점']==156].copy()
wthr['dt']=pd.to_datetime(wthr['일시'])
wx=wthr[['dt','기온(°C)','습도(%)','풍속(m/s)']].copy()
wx.columns=['dt','temp','humid','wind_ms']
for c in ['temp','humid','wind_ms']: wx[c]=wx[c].interpolate().ffill().bfill()

# ═══ 2. 결합 & 분해 ═══
df=solar.merge(wind,on='dt',how='outer').merge(ll,on='dt',how='inner').merge(wx,on='dt',how='left')
df=df.sort_values('dt').reset_index(drop=True)
df['solar']=df['solar'].fillna(0); df['wind']=df['wind'].fillna(0)
df['ym']=df['dt'].dt.to_period('M'); df['year']=df['dt'].dt.year; df['month']=df['dt'].dt.month
df=df.merge(jr,on=['year','month'],how='left'); df=df[df['ratio'].notna()].copy()
df['mkt']=df['nat']*df['ratio']
df=df.join(cap_m,on='ym')
df['s']=(df['solar']/df['태양_계량']).clip(lower=0,upper=1)
df['K']=df['자가용']
df['unmetered']=df['K']*df['s']
df['metered']=df['solar']+df['wind']
df['total_0']=df['mkt']+df['unmetered']
df=df.dropna(subset=['total_0','temp']).reset_index(drop=True)

# ═══ 3. 총수요 N-BEATS (기온·달력만) ═══
df['hour']=df['dt'].dt.hour; df['dow']=df['dt'].dt.dayofweek
df['is_weekend']=(df['dow']>=5).astype(int)
kr=holidays.SouthKorea(years=[2023,2024,2025])
df['is_holiday']=df['dt'].dt.date.astype('datetime64[ns]').isin(pd.to_datetime(list(kr.keys()))).astype(int)

# ── N-BEATS ──
class Block(nn.Module):
    def __init__(s,i,th,fc,h,b='g'):
        super().__init__(); s.i,s.h,s.b=i,h,b
        s.fc=nn.Sequential(nn.Linear(i,fc),nn.ReLU(),nn.Linear(fc,fc),nn.ReLU(),
            nn.Linear(fc,fc),nn.ReLU(),nn.Linear(fc,th))
        if b=='t':
            d=th//2; tb=torch.arange(i).float()/i; tf=torch.arange(h).float()/h
            s.register_buffer('Tb',torch.stack([tb**k for k in range(d)]))
            s.register_buffer('Tf',torch.stack([tf**k for k in range(d)])); s.d=d
        elif b=='s':
            hh=th//4; tb=2*np.pi*np.arange(i)/i; tf=2*np.pi*np.arange(h)/h; fr=np.arange(1,hh+1)
            s.register_buffer('Sb',torch.FloatTensor(np.concatenate([np.cos(np.outer(fr,tb)),np.sin(np.outer(fr,tb))])))
            s.register_buffer('Sf',torch.FloatTensor(np.concatenate([np.cos(np.outer(fr,tf)),np.sin(np.outer(fr,tf))]))); s.hh=hh
    def forward(s,x):
        t=s.fc(x)
        if s.b=='g': return t[:,:s.i],t[:,s.i:]
        if s.b=='t': return t[:,:s.d]@s.Tb, t[:,s.d:]@s.Tf
        return t[:,:2*s.hh]@s.Sb, t[:,2*s.hh:]@s.Sf
class Stack(nn.Module):
    def __init__(s,i,h,fc,n,th,b):
        super().__init__(); s.bl=nn.ModuleList([Block(i,th,fc,h,b) for _ in range(n)]); s.h=h
    def forward(s,x):
        r=x; f=torch.zeros(x.size(0),s.h,device=x.device)
        for b in s.bl: bc,ff=b(r); r=r-bc; f=f+ff
        return r,f
class NBeats(nn.Module):
    def __init__(s,i=LOOKBACK,h=HORIZON,fc=256,n=3):
        super().__init__()
        s.g=Stack(i,h,fc,n,i+h,'g'); s.t=Stack(i,h,fc,n,8,'t'); s.s=Stack(i,h,fc,n,h*2,'s')
    def forward(s,x):
        r0,f0=s.g(x); r1,f1=s.t(r0); _,f2=s.s(r1); return f0+f1+f2
class DS(Dataset):
    def __init__(s,ser,lb=LOOKBACK,hz=HORIZON):
        s.x,s.y=[],[]
        for i in range(lb,len(ser)-hz+1):
            s.x.append(ser[i-lb:i]); s.y.append(ser[i:i+hz])
        s.x=torch.FloatTensor(np.array(s.x)); s.y=torch.FloatTensor(np.array(s.y))
    def __len__(s): return len(s.x)
    def __getitem__(s,i): return s.x[i],s.y[i]

tr=df[df['year']<=2024].reset_index(drop=True)
va=df[df['year']==2025].reset_index(drop=True)
TARGET='total_0'
sc=StandardScaler(); sc.fit(tr[[TARGET]].values)
tr_s=sc.transform(tr[[TARGET]].values).flatten()
va_full=np.concatenate([tr[TARGET].values[-LOOKBACK:], va[TARGET].values])
va_s=sc.transform(va_full.reshape(-1,1)).flatten()
tr_dl=DataLoader(DS(tr_s),batch_size=64,shuffle=True,drop_last=True)
va_dl=DataLoader(DS(va_s),batch_size=64,shuffle=False)

dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=NBeats().to(dev)
opt=torch.optim.Adam(model.parameters(),lr=8e-4,weight_decay=1e-5)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
lf=nn.HuberLoss(delta=1.0)

print("="*55); print("  전남 총수요·비계량 (형 방법 · N-BEATS)"); print("="*55)
print(f"학습 {len(tr):,}h (2023~24) | 검증 {len(va):,}h (2025)")
print(f"K_r(비계량설비) 평균: {df['K'].mean():.1f} MW | s 정오평균: {df[df['hour']==12]['s'].mean():.3f}\n[학습]")

for ep in range(1,EPOCHS+1):
    model.train(); t=0
    for xb,yb in tr_dl:
        xb,yb=xb.to(dev),yb.to(dev); opt.zero_grad()
        l=lf(model(xb),yb); l.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); t+=l.item()*len(xb)
    sch.step()
    if ep%5==0: print(f"  Epoch {ep}/{EPOCHS} Train={t/len(tr_dl.dataset):.4f}")

model.eval(); P,T=[],[]
with torch.no_grad():
    for xb,yb in va_dl: P.append(model(xb.to(dev)).cpu().numpy()); T.append(yb.numpy())
P=np.concatenate(P); T=np.concatenate(T)
pi=sc.inverse_transform(P.reshape(-1,1)).reshape(P.shape)
ti=sc.inverse_transform(T.reshape(-1,1)).reshape(T.shape)
mae=mean_absolute_error(ti.flatten(),pi.flatten())
rmse=np.sqrt(mean_squared_error(ti.flatten(),pi.flatten()))
mape=np.mean(np.abs((ti-pi)/ti))*100
print(f"\n[총수요 예측] MAE={mae:.1f} RMSE={rmse:.1f} MAPE={mape:.2f}%")

# 비계량 검증 (KPX 추계 대조)
uk=pd.read_csv('solar_unmetered_hourly_2023_2025.csv',encoding='utf-8-sig')
uk['dt']=pd.to_datetime(uk['datetime'])
va_v=va.merge(uk[['dt','비계량태양광_합계']],on='dt',how='left')
va_v['kpx_jn']=va_v['비계량태양광_합계']*va_v['ratio']
day=va_v[va_v['s']>0.05]
print(f"[비계량 검증] 우리 vs KPX추계×비율 상관: {day['unmetered'].corr(day['kpx_jn']):.3f}")
print(f"  우리 비계량합: {va_v['unmetered'].sum():.0f} | KPX추계×비율합: {va_v['kpx_jn'].sum():.0f} MWh")

# 저장 (검증기간 시간별)
tt=va['dt'].values[LOOKBACK-LOOKBACK:]  # va 전체
rows=[]
for i in range(ti.shape[0]):
    for h in range(HORIZON):
        idx=i+h
        if idx<len(va):
            rows.append({'시간':va['dt'].iloc[idx],'총수요_실제':round(ti[i,h],2),'총수요_예측':round(pi[i,h],2),
                         '비계량':round(va['unmetered'].iloc[idx],2)})
out=pd.DataFrame(rows).drop_duplicates('시간',keep='last').sort_values('시간')
out.to_csv('전남_Nbeats_결과.csv',index=False,encoding='utf-8-sig')
print("\n[저장] 전남_Nbeats_결과.csv")
