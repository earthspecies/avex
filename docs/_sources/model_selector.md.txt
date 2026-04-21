# Model Selector

```{raw} html

<style>
.ms-divider{border:none;border-top:.5px solid var(--color-background-border);margin-bottom:1.5rem}

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --eg:#1D9E75;--egl:#E1F5EE;--egm:#5DCAA5;--eat:#0F6E56;
  --ea:#EF9F27;--eal:#FAEEDA;--eco:#D85A30;--ecl:#FAECE7;
  --eb:#185FA5;--ebl:#E6F1FB;--ep:#8B5CF6;--epl:#F0EFFE;
  --fm:'Euclid Circular B',sans-serif;--mo:'JetBrains Mono',monospace
}
.wr{font-family:var(--fm);padding:0 0 2rem;color:var(--color-foreground-primary)}
.hdr{border-bottom:.5px solid var(--color-background-border);padding:1.25rem 0 1rem;margin-bottom:1.5rem}
.hdr h1{font-size:16px;font-weight:500;letter-spacing:-.01em;display:flex;align-items:center;gap:8px}
.hdr p{font-size:14px;color:var(--color-foreground-secondary);margin-top:4px;line-height:1.5}
.badge{display:inline-flex;align-items:center;gap:5px;font-size:12px;font-weight:500;padding:3px 8px;background:var(--egl);color:var(--eat);border-radius:20px;border:.5px solid var(--egm);font-family:var(--mo)}
.dot{width:6px;height:6px;background:var(--eg);border-radius:50%;display:inline-block}
.pb{display:flex;align-items:center;justify-content:space-between;margin-bottom:1.5rem;width:100%}
.ps{display:flex;align-items:center;gap:6px;font-size:13px;color:var(--color-foreground-muted);transition:color .2s}
.ps.active{color:var(--eg);font-weight:500}.ps.done{color:var(--color-foreground-secondary)}
.pd{width:20px;height:20px;border-radius:50%;background:var(--color-background-secondary);border:.5px solid var(--color-background-border);display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:500;transition:all .2s;flex-shrink:0}
.ps.active .pd{background:var(--eg);border-color:var(--eg);color:#fff}
.ps.done .pd{background:var(--egl);border-color:var(--egm);color:var(--eat)}
.pc{flex:1;height:.5px;background:var(--color-background-border);min-width:12px}
.qp{background:var(--color-background-secondary);border-radius:0.5rem;border:.5px solid var(--color-background-border);padding:1rem;margin-bottom:1rem}
.ql{font-size:12px;font-weight:500;letter-spacing:.06em;text-transform:uppercase;color:var(--color-foreground-muted);margin-bottom:6px;font-family:var(--mo)}
.qt{font-size:16px;font-weight:500;line-height:1.4;margin-bottom:4px}
.qh{font-size:13px;color:var(--color-foreground-secondary);line-height:1.5}
.og{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-bottom:1rem}
.ob{background:var(--color-background-primary);border:.5px solid var(--color-background-border);border-radius:0.35rem;padding:.75rem 1rem .75rem 1.1rem;cursor:pointer;text-align:left;transition:all .15s;display:flex;flex-direction:column;gap:4px;position:relative;overflow:hidden}
.ob::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:transparent;transition:background .15s}
.ob:hover{border-color:var(--egm);background:var(--egl)}.ob:hover::before{background:var(--eg)}
.oi{font-size:18px;line-height:1;margin-bottom:2px}.ott{font-size:14px;font-weight:500;line-height:1.3}
.od{font-size:12px;color:var(--color-foreground-secondary);line-height:1.4}
.ct{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:1rem}
.cr{display:flex;align-items:center;gap:5px;font-size:12px;font-weight:500;padding:3px 9px;background:var(--egl);color:var(--eat);border-radius:20px;border:.5px solid var(--egm);cursor:pointer;transition:opacity .15s}
.cr:hover{opacity:.7}.cx{font-size:12px;opacity:.6}
.rh{margin-bottom:1.5rem}.rt{font-size:15px;font-weight:500;margin-bottom:4px}.rs{font-size:13px;color:var(--color-foreground-secondary)}
.mc{background:var(--color-background-primary);border:.5px solid var(--color-background-border);border-radius:0.5rem;padding:1.25rem 1.5rem;margin-bottom:12px;overflow:hidden}
.mc.tp{border:1.5px solid var(--eg)}
.tpbanner{background:var(--egl);color:var(--eat);font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;text-align:center;padding:4px 0;margin:-1.25rem -1.5rem 1.25rem;border-bottom:1px solid var(--egm)}
.mch{display:flex;align-items:flex-start;gap:12px;margin-bottom:14px}
.mr{width:24px;height:24px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:500;flex-shrink:0}
.r1{background:var(--egl);color:var(--eat);border:.5px solid var(--egm)}
.r2{background:var(--eal);color:#854F0B;border:.5px solid #FAC775}
.r3{background:var(--color-background-secondary);color:var(--color-foreground-secondary);border:.5px solid var(--color-background-border)}
.mnw{flex:1;min-width:0}.mn{font-size:14px;font-weight:500;margin-bottom:2px}
.mid{font-size:12px;font-family:var(--mo);color:var(--color-foreground-secondary)}
.tpb{font-size:12px;font-weight:500;padding:2px 7px;background:var(--egl);color:var(--eat);border-radius:20px;border:.5px solid var(--egm);white-space:nowrap;flex-shrink:0}
.md{font-size:13px;color:var(--color-foreground-secondary);line-height:1.6;margin-bottom:12px}
.mt{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:14px}
.tag{font-size:12px;font-weight:500;padding:2px 7px;border-radius:20px}
.ta{background:var(--ebl);color:var(--eb);border:.5px solid #B5D4F4}
.td{background:var(--eal);color:#854F0B;border:.5px solid #FAC775}
.tk{background:var(--egl);color:var(--eat);border:.5px solid var(--egm)}
.tp2{background:var(--ecl);color:#993C1D;border:.5px solid #F5C4B3}
.sg{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px}
.sl2{font-size:12px;color:var(--color-foreground-muted);margin-bottom:3px;font-family:var(--mo)}
.sbw{display:flex;align-items:center;gap:6px}
.sb{flex:1;height:4px;background:var(--color-background-secondary);border-radius:2px;overflow:hidden}
.sf{height:100%;border-radius:2px;transition:width .6s ease}
.sf.g{background:var(--eg)}.sf.a{background:var(--ea)}.sf.c{background:var(--eco)}.sf.b{background:var(--eb)}
.sv{font-size:12px;font-weight:500;font-family:var(--mo);color:var(--color-foreground-secondary);min-width:28px;text-align:right}
.pg{border:.5px solid var(--color-background-border);border-radius:0.35rem;padding:1rem 1.1rem;margin-bottom:10px;background:var(--color-background-secondary)}
.pgt{font-size:13px;font-weight:500;color:var(--color-foreground-primary);margin-bottom:6px;margin-top:12px;display:flex;align-items:center;gap:6px}
.pgr{display:flex;flex-direction:column;gap:0}
.pgc{display:flex;flex-direction:column;gap:4px;padding:14px 0;border-top:.5px solid var(--color-background-border)}.pgc:first-child{padding-top:0;border-top:none}.pgc:last-child{padding-bottom:0}
.pgl{font-size:12px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;color:var(--color-foreground-muted);font-family:var(--mo);margin-bottom:2px}
.pgn{font-size:12px;color:var(--color-foreground-secondary);line-height:1.4}
.pill{display:inline-flex;align-items:center;gap:4px;font-size:12px;font-weight:500;padding:2px 7px;border-radius:20px}
.pill.g{background:var(--egl);color:var(--eat);border:.5px solid var(--egm)}
.pill.a{background:var(--eal);color:#854F0B;border:.5px solid #FAC775}
.pill.p{background:var(--epl);color:#5B21B6;border:.5px solid #C4B5FD}
.pill.b{background:var(--ebl);color:var(--eb);border:.5px solid #B5D4F4}
.ml{display:inline-flex;align-items:center;gap:5px;font-size:12px;font-weight:500;color:var(--eg);text-decoration:none;padding:4px 10px;border:.5px solid var(--egm);border-radius:0.35rem;background:var(--egl);transition:opacity .15s}
.ml:hover{opacity:.75}
.lc{font-family:var(--mo);font-size:12px;background:var(--color-background-secondary);color:var(--color-foreground-secondary);padding:16px 20px;border-radius:0.35rem;border:.5px solid var(--color-background-border);display:block;margin-top:16px;overflow-x:auto;white-space:pre}
.mf{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:0;justify-content:flex-end}
.nr{display:flex;align-items:center;gap:8px;margin-top:1rem}
.bk{background:transparent;border:.5px solid var(--color-background-border);border-radius:0.35rem;padding:6px 14px;font-size:13px;cursor:pointer;color:var(--color-foreground-secondary);transition:all .15s;font-family:var(--fm)}
.bk:hover{background:var(--color-background-secondary);color:var(--color-foreground-primary)}
.br{background:transparent;border:none;font-size:13px;cursor:pointer;color:var(--eg);transition:color .15s;font-family:var(--fm);padding:6px 0;font-weight:500}
.br:hover{opacity:.7}
.ct2{font-size:12px;color:var(--color-foreground-secondary);cursor:pointer;background:none;border:none;font-family:var(--fm);padding:4px 0;margin-top:.5rem;display:block;transition:color .15s}
.ct2:hover{color:var(--eg)}
.ctb{width:100%;font-size:12px;border-collapse:collapse;margin-top:8px}
.ctb th{text-align:left;padding:4px 8px;font-weight:500;color:var(--color-foreground-secondary);border-bottom:.5px solid var(--color-background-border);font-family:var(--mo);font-size:12px;background:var(--color-background-secondary)}
.ctb td{padding:5px 8px;color:var(--color-foreground-secondary);border-bottom:.5px solid var(--color-background-border);font-family:var(--mo);font-size:12px}
.ctb tr:last-child td{border-bottom:none}.ctb tr td:first-child{color:var(--color-foreground-primary);font-weight:500}
.vg{color:var(--eat)!important}.vm{color:#854F0B!important}
.pn{font-size:12px;color:var(--color-foreground-muted);margin-top:1rem;padding-top:1rem;border-top:.5px solid var(--color-background-border);display:flex;align-items:center;gap:6px;line-height:1.5}
.pn a{color:var(--eg);text-decoration:none}.pn a:hover{text-decoration:underline}
.ld{margin-top:8px;padding:8px 10px;background:var(--color-background-secondary);border-radius:0.35rem;border:.5px solid var(--color-background-border)}
.ldt{font-size:12px;font-weight:500;color:var(--color-foreground-muted);font-family:var(--mo);margin-bottom:6px;letter-spacing:.05em;text-transform:uppercase}
.lbr{display:flex;align-items:center;gap:4px;margin-bottom:4px}
.lb{height:12px;border-radius:2px;font-size:11px;display:flex;align-items:center;padding:0 5px;font-weight:500;font-family:var(--mo);white-space:nowrap;transition:all .3s}
.lb.hi{background:var(--eg);color:#fff}
.lb.mid2{background:var(--ea);color:#fff}
.lb.lo{background:var(--eb);color:#fff}
.lb.dim{background:var(--color-background-primary);color:var(--color-foreground-muted);border:.5px solid var(--color-background-border)}
.ln{font-size:11px;color:var(--color-foreground-muted);font-family:var(--mo);min-width:44px}
</style>

<p id="ms-intro" style="font-size:0.88rem;color:var(--color-foreground-secondary);line-height:1.6;margin-bottom:1.25rem">Not sure which model to use? Answer a few questions about your task and data, and we'll recommend the right AVES-2 model and probing strategy.</p>
<div class="wr">
<div id="app"></div>
</div>

<script>
const MODELS={
  sl_beats_all:{id:'esp_aves2_sl_beats_all',name:'SL-BEATs-All',hf:'https://huggingface.co/EarthSpeciesProject/esp-aves2-sl-beats-all',arch:'BEATs',training:'SL',data:'AudioSet + Bio',desc:'Best overall encoder. SSL pre-trained BEATs followed by supervised post-training on the full mixed corpus. Strongest in- and out-of-distribution generalisation across all 26 datasets.',tags:[{l:'BEATs',c:'ta'},{l:'All data',c:'td'},{l:'Best overall',c:'tp2'}],scores:{'BEANS cls.':83.2,'BEANS det.':40.8,'Indiv. ID':51.1,'Vocal rep.':79.8}},
  sl_beats_bio:{id:'esp_aves2_sl_beats_bio',name:'SL-BEATs-Bio',hf:'https://huggingface.co/EarthSpeciesProject/esp-aves2-sl-beats-bio',arch:'BEATs',training:'SL',data:'Bioacoustics only',desc:'BEATs fine-tuned purely on bioacoustic data. Strong on bird species tasks. May outperform the All-data model on narrow, in-distribution species tasks.',tags:[{l:'BEATs',c:'ta'},{l:'Bio only',c:'td'},{l:'Specialist',c:'tk'}],scores:{'BEANS cls.':81.0,'BEANS det.':38.4,'Indiv. ID':49.2,'Vocal rep.':77.1}},
  naturelm_beats:{id:'esp_aves2_naturelm_audio_v1_beats',name:'NatureLM-Audio BEATs',hf:'https://huggingface.co/EarthSpeciesProject/esp-aves2-naturelm-audio-v1-beats',arch:'BEATs',training:'SL',data:'NatureLM-Audio (Audio-LM)',desc:'BEATs backbone extracted from NatureLM-Audio, unfrozen during large-scale audio-language training. Strong zero-shot and cross-taxa transfer.',tags:[{l:'BEATs',c:'ta'},{l:'Audio-LM derived',c:'td'},{l:'Zero-shot',c:'tk'}],scores:{'BEANS cls.':80.5,'BEANS det.':38.1,'Indiv. ID':47.8,'Vocal rep.':76.4}},
  eat_all:{id:'esp_aves2_eat_all',name:'SL-EAT-All',hf:'https://huggingface.co/EarthSpeciesProject/esp-aves2-eat-all',arch:'EAT',training:'SL',data:'AudioSet + Bio',desc:'Efficient Audio Transformer on full mixed corpus. Competitive with BEATs at a smaller footprint. Good when GPU budget is moderate.',tags:[{l:'EAT',c:'ta'},{l:'All data',c:'td'},{l:'Efficient',c:'tp2'}],scores:{'BEANS cls.':79.3,'BEANS det.':36.8,'Indiv. ID':46.5,'Vocal rep.':74.2}},
  eat_bio:{id:'esp_aves2_eat_bio',name:'SL-EAT-Bio',hf:'https://huggingface.co/EarthSpeciesProject/esp-aves2-eat-bio',arch:'EAT',training:'SL',data:'Bioacoustics only',desc:'EAT fine-tuned on bio data only. Efficient and domain-focused. Good for edge or batch deployments targeting a specific taxon group.',tags:[{l:'EAT',c:'ta'},{l:'Bio only',c:'td'},{l:'Efficient',c:'tp2'}],scores:{'BEANS cls.':77.1,'BEANS det.':34.5,'Indiv. ID':44.0,'Vocal rep.':72.8}},
  sl_eat_all_ssl_all:{id:'esp_aves2_sl_eat_all_ssl_all',name:'SSL→SL EAT-All',hf:'https://huggingface.co/EarthSpeciesProject/esp-aves2-sl-eat-all-ssl-all',arch:'EAT',training:'SSL→SL',data:'SSL on All → SL on All',desc:'Full two-stage recipe: self-supervised pre-training on mixed corpus, then supervised post-training. Excellent out-of-distribution generalisation.',tags:[{l:'EAT',c:'ta'},{l:'SSL→SL',c:'td'},{l:'OOD robust',c:'tp2'}],scores:{'BEANS cls.':80.1,'BEANS det.':37.5,'Indiv. ID':47.2,'Vocal rep.':75.6}},
  sl_eat_bio_ssl_all:{id:'esp_aves2_sl_eat_bio_ssl_all',name:'SSL→SL EAT-Bio',hf:'https://huggingface.co/EarthSpeciesProject/esp-aves2-sl-eat-bio-ssl-all',arch:'EAT',training:'SSL→SL',data:'SSL on All → SL on Bio',desc:'SSL pre-trained on diverse audio then specialised on bioacoustics. Excellent OOD performance with limited data. Ideal for understudied non-bird taxa.',tags:[{l:'EAT',c:'ta'},{l:'SSL→SL Bio',c:'td'},{l:'Limited data',c:'tk'}],scores:{'BEANS cls.':78.9,'BEANS det.':36.2,'Indiv. ID':46.0,'Vocal rep.':73.9}},
  effnetb0_all:{id:'esp_aves2_effnetb0_all',name:'EfficientNet-B0-All',hf:'https://huggingface.co/EarthSpeciesProject/esp-aves2-effnetb0-all',arch:'EfficientNet-B0',training:'SL',data:'AudioSet + Bio',desc:'Lightweight CNN on all data. Much faster inference than transformers. Best for high-throughput or resource-constrained deployments.',tags:[{l:'CNN',c:'ta'},{l:'All data',c:'td'},{l:'Fast inference',c:'tp2'}],scores:{'BEANS cls.':74.5,'BEANS det.':31.2,'Indiv. ID':41.3,'Vocal rep.':69.1}},
  effnetb0_bio:{id:'esp_aves2_effnetb0_bio',name:'EfficientNet-B0-Bio',hf:'https://huggingface.co/EarthSpeciesProject/esp-aves2-effnetb0-bio',arch:'EfficientNet-B0',training:'SL',data:'Bioacoustics only',desc:'Lightweight CNN focused on bioacoustics. Fast and compact. Good for on-device monitoring of a known taxon.',tags:[{l:'CNN',c:'ta'},{l:'Bio only',c:'td'},{l:'Edge/on-device',c:'tk'}],scores:{'BEANS cls.':72.8,'BEANS det.':29.7,'Indiv. ID':39.4,'Vocal rep.':67.5}},
  effnetb0_audioset:{id:'esp_aves2_effnetb0_audioset',name:'EfficientNet-B0-AudioSet',hf:'https://huggingface.co/EarthSpeciesProject/esp-aves2-effnetb0-audioset',arch:'EfficientNet-B0',training:'SL',data:'AudioSet only',desc:'Baseline CNN on AudioSet. Useful when general audio understanding matters and bioacoustic specialisation is not required.',tags:[{l:'CNN',c:'ta'},{l:'AudioSet',c:'td'},{l:'Baseline',c:'tp2'}],scores:{'BEANS cls.':71.2,'BEANS det.':28.4,'Indiv. ID':37.8,'Vocal rep.':65.3}}
};

function getProbingGuidance(modelId,task,taxon,ood){
  const m=MODELS[modelId];if(!m)return null;
  const isSSL=m.training==='SSL→SL';
  const isCNN=m.arch==='EfficientNet-B0';
  const isBirdSpecies=(task==='species'||task==='detection')&&(taxon==='birds'||taxon==='terrestrial')&&!ood;
  let probeType,layerRec,layerRationale,probeRationale,costNote,layers;
  if(isCNN){
    probeType='linear';layerRec='all-layer (via adapters)';
    layerRationale='All 15 blocks via adapters. CNNs treat spectrograms as 2D images with no explicit temporal structure. All-layer probing adds ~+0.09 accuracy on BEANS cls. despite large adapter cost.';
    probeRationale='Attention probes do not improve EfficientNet — no temporal dependencies to exploit. Use linear probe with all-layer extraction.';
    costNote='All-layer on EfficientNet requires large adapters (~41–105M trainable params). Use last-layer as a fast baseline if compute is tight.';
    layers=[{n:'L1–5',cls:'mid2'},{n:'L6–10',cls:'mid2'},{n:'L11–15',cls:'mid2'}];
  } else if(isSSL){
    probeType='attention';layerRec='all-layer (middle layers weighted highest)';
    layerRationale='SSL backbones encode rich temporal structure across layers, typically peaking in middle blocks (L4–7). Multi-layer probing with learned softmax weights exploits this. All-layer + attention: ~+0.08 accuracy over last-layer linear.';
    probeRationale='SSL transformers model temporal dependencies explicitly — attention probes fully exploit this. Pairing an SSL backbone with an attention head is the strongest transfer setup.';
    costNote='2.40M params for attention probe. All-layer adds adapter alignment handled automatically in AVEX.';
    layers=[{n:'L1–3',cls:'lo'},{n:'L4–7 ★',cls:'hi'},{n:'L8–11',cls:'mid2'}];
  } else if(isBirdSpecies){
    probeType='linear (last or all)';layerRec='last layer or upper layers';
    layerRationale='SL models trained on bioacoustics + AudioSet specialise bird species into their upper/last layers. For in-distribution bird tasks the final layer is already well-calibrated. All-layer still gives ~+0.05 accuracy gain.';
    probeRationale='For in-distribution bird species tasks with SL models, linear probes on upper layers are a strong, efficient baseline. Attention probes add modest marginal gains.';
    costNote='37.68K params for linear, 2.40M for attention — both negligible vs the 90M frozen backbone.';
    layers=[{n:'L1–4',cls:'dim'},{n:'L5–8',cls:'lo'},{n:'L9–11 ★',cls:'hi'}];
  } else {
    probeType='attention';layerRec='all-layer (lower–middle layers most informative)';
    layerRationale='For tasks or taxa distant from the SL training domain (non-birds, individual ID, vocal repertoire), lower and middle layers capture more general acoustic structure. Learned softmax weights in multi-layer probing automatically up-weight these. Gains: ~+0.08 accuracy over last-layer linear.';
    probeRationale='SL transformers on non-bird or OOD tasks benefit most from attention probes + multi-layer extraction. Task-relevant structure lives in early–mid layers, not the specialised final layer.';
    costNote='37.68K params for linear, 2.40M for attention — both negligible vs the 90M frozen backbone.';
    layers=[{n:'L1–3 ★',cls:'hi'},{n:'L4–7 ★',cls:'hi'},{n:'L8–11',cls:'lo'}];
  }
  return{probeType,layerRec,layerRationale,probeRationale,costNote,layers,isCNN,isSSL,isBirdSpecies};
}

function renderLayerDiagram(layers){
  const legend={hi:'high',mid2:'medium',lo:'lower',dim:'minimal'};
  let html='';
  for(const l of layers){
    const w=l.cls==='hi'?'100%':l.cls==='mid2'?'65%':l.cls==='lo'?'40%':'20%';
    const label=l.n.includes('★')?'★ most useful':legend[l.cls];
    html+=`<div class="lbr"><span class="ln">${l.n.replace(' ★','')}</span><div class="lb ${l.cls}" style="width:${w}">${label}</div></div>`;
  }
  return html;
}

const TREE={
  id:'root',q:'What is your primary task?',hint:'If you don\'t see your task here, pick the closest match.',
  options:[
    {id:'species',icon:'🦜',title:'Species classification',desc:'Identify species in audio recordings',next:{
      id:'species_data',q:'What does your data look like?',hint:'',
      options:[
        {id:'species_data_diverse',icon:'🌍',title:'Diverse / multi-taxa',desc:'Multiple taxa, varied environments',next:{
          id:'species_compute',q:'Inference compute budget?',hint:'',
          options:[
            {id:'compute_any',icon:'⚡',title:'No constraint',desc:'',result:['sl_beats_all','eat_all','sl_eat_all_ssl_all'],taxon:'diverse',task:'species'},
            {id:'compute_low',icon:'🔋',title:'Low / edge / batch',desc:'',result:['effnetb0_all','eat_all'],taxon:'diverse',task:'species'}
          ]
        }},
        {id:'species_data_bio',icon:'🧬',title:'Bioacoustics focused',desc:'Bird-rich or single taxonomic group',next:{
          id:'species_ood',q:'Will you evaluate on unseen species or soundscapes?',hint:'',
          options:[
            {id:'ood_yes',icon:'🗺️',title:'Yes — new species / habitats',desc:'Generalisation is critical',result:['sl_beats_all','sl_eat_all_ssl_all','sl_beats_bio'],taxon:'ood',task:'species',ood:true},
            {id:'ood_no',icon:'🎯',title:'No — same domain, closed set',desc:'Known species list, focal recordings',result:['sl_beats_bio','eat_bio','sl_eat_bio_ssl_all'],taxon:'birds',task:'species',ood:false}
          ]
        }},
        {id:'species_data_limited',icon:'🔬',title:'Very limited labels',desc:'Few annotations, understudied taxon',result:['sl_beats_all','sl_eat_all_ssl_all','naturelm_beats'],taxon:'diverse',task:'species',ood:true}
      ]
    }},
    {id:'individual',icon:'🐬',title:'Individual identification',desc:'Match vocalisations to specific individuals',next:{
      id:'individual_taxon',q:'Which taxonomic group?',hint:'',
      options:[
        {id:'ind_birds',icon:'🐦',title:'Birds',desc:'Avian individual ID',result:['sl_beats_all','sl_beats_bio','sl_eat_all_ssl_all'],taxon:'birds_indiv',task:'individual',ood:false},
        {id:'ind_mammals',icon:'🐋',title:'Marine / terrestrial mammals',desc:'Cetaceans, primates, bats, etc.',result:['sl_beats_all','naturelm_beats','sl_eat_all_ssl_all'],taxon:'mammals',task:'individual',ood:true},
        {id:'ind_other',icon:'🦗',title:'Other / understudied taxa',desc:'Insects, amphibians, fish, etc.',result:['sl_beats_all','sl_eat_all_ssl_all','eat_all'],taxon:'other',task:'individual',ood:true}
      ]
    }},
    {id:'repertoire',icon:'🎵',title:'Vocal repertoire discovery',desc:'Cluster or retrieve vocalisation types (unsupervised)',next:{
      id:'repertoire_mode',q:'What kind of analysis?',hint:'',
      options:[
        {id:'rep_retrieval',icon:'🔍',title:'Retrieval / similarity search',desc:'Find vocalisations similar to a query',result:['sl_beats_all','sl_eat_all_ssl_all','naturelm_beats'],taxon:'other',task:'repertoire',ood:true},
        {id:'rep_clustering',icon:'🗂️',title:'Clustering / call type discovery',desc:'Unsupervised grouping of call types',result:['sl_beats_all','sl_beats_bio','sl_eat_all_ssl_all'],taxon:'other',task:'repertoire',ood:true}
      ]
    }},
    {id:'embedding',icon:'📐',title:'General embeddings / transfer',desc:'Extract features for a custom downstream model',next:{
      id:'embed_priority',q:'What matters most for your project?',hint:'',
      options:[
        {id:'embed_perf',icon:'🏆',title:'Highest quality embeddings',desc:'Best linear probing benchmark scores',result:['sl_beats_all','sl_eat_all_ssl_all','eat_all'],taxon:'diverse',task:'embedding',ood:true},
        {id:'embed_speed',icon:'🚀',title:'Fast inference / lightweight',desc:'High-throughput or on-device',result:['effnetb0_all','eat_bio','eat_all'],taxon:'diverse',task:'embedding',ood:false},
        {id:'embed_zeroshot',icon:'✨',title:'Zero/few-shot transfer',desc:'New taxa with minimal finetuning',result:['naturelm_beats','sl_beats_all','sl_eat_all_ssl_all'],taxon:'other',task:'embedding',ood:true}
      ]
    }},
    {id:'detection',icon:'🔊',title:'Sound event detection (soundscapes)',desc:'Detect calls in long field recordings or soundscapes',next:{
      id:'det_type',q:'Target acoustic environment?',hint:'',
      options:[
        {id:'det_terrestrial',icon:'🌳',title:'Terrestrial / forest soundscapes',desc:'Birds, amphibians, insects in acoustic scene',result:['sl_beats_all','sl_eat_all_ssl_all','eat_all'],taxon:'terrestrial',task:'detection',ood:true},
        {id:'det_marine',icon:'🌊',title:'Marine / underwater',desc:'Hydrophone recordings, cetaceans',result:['sl_beats_all','naturelm_beats','sl_eat_all_ssl_all'],taxon:'mammals',task:'detection',ood:true},
        {id:'det_edge',icon:'📡',title:'Real-time / edge monitoring',desc:'On-device sensors, low-power hardware',result:['effnetb0_all','effnetb0_bio','eat_bio'],taxon:'diverse',task:'detection',ood:false}
      ]
    }}
  ]
};

const scoreColors={'BEANS cls.':'g','BEANS det.':'a','Indiv. ID':'c','Vocal rep.':'b'};
const maxScores={'BEANS cls.':100,'BEANS det.':60,'Indiv. ID':70,'Vocal rep.':100};
let state={path:[],selections:[],showCompare:true};

function getLeaf(){
  let n=TREE;
  for(const id of state.path){
    const opt=n&&n.options&&n.options.find(o=>o.id===id);
    if(!opt)return null;
    if(opt.result)return opt;
    n=opt.next;
  }
  return null;
}

function getCurrentNode(){
  let n=TREE;
  for(const id of state.path){
    const opt=n&&n.options&&n.options.find(o=>o.id===id);
    if(!opt)return null;
    if(opt.result)return null;
    n=opt.next;
  }
  return n;
}

function getSteps(){
  let steps=[],n=TREE,depth=0;
  while(n&&n.options){
    steps.push({q:n.q,depth});
    const sel=state.path[depth];
    if(sel===undefined)break;
    const opt=n.options.find(o=>o.id===sel);
    if(!opt)break;
    if(opt.result)break;
    n=opt.next;depth++;
  }
  return steps;
}

function renderProgress(){
  const steps=getSteps();
  let html='<div class="pb">';
  for(let i=0;i<Math.max(steps.length,1);i++){
    if(i>0)html+='<div class="pc"></div>';
    const done=i<state.path.length,active=i===state.path.length&&!getLeaf();
    const cls=done?'done':active?'active':'';
    const label=steps[i]?steps[i].q.split(' ').slice(0,4).join(' ')+'…':'';
    html+=`<div class="ps ${cls}"><div class="pd">${i+1}</div></div>`;
  }
  return html+'</div>';
}

function renderCrumbs(){
  if(!state.selections.length)return'';
  let html='<div class="ct">';
  state.selections.forEach((s,i)=>html+=`<div class="cr" onclick="goTo(${i})"><span>${s}</span><span class="cx">×</span></div>`);
  return html+'</div>';
}

function renderNavRow(showBack){
  if(!showBack)return'';
  return`<div class="nr"><button class="bk" onclick="goBack()">← Back</button><button class="br" onclick="resetAll()">Start over</button></div>`;
}

function renderQuestion(node){
  let html=`<div class="qp"><div class="qt">${node.q}</div>${node.hint?`<div class="qh">${node.hint}</div>`:''}</div><div class="og">`;
  for(const opt of node.options){
    html+=`<button class="ob" onclick="selectOption('${opt.id}','${opt.title}')"><div class="oi">${opt.icon}</div><div class="ott">${opt.title}</div><div class="od">${opt.desc}</div></button>`;
  }
  html+='</div>';
  html+=renderNavRow(state.path.length>0);
  return html;
}

function renderProbeGuidance(modelId,leaf){
  const task=leaf.task||'embedding',taxon=leaf.taxon||'diverse',ood=leaf.ood||false;
  const g=getProbingGuidance(modelId,task,taxon,ood);
  if(!g)return'';
  return`<div class="pgt">Recommended probing strategy</div>
  <div class="pg">
    <div class="pgr">
      <div class="pgc">
        <div class="pgl">Probe type</div>
        <div style="margin-bottom:4px"><span class="pill ${g.isCNN?'b':g.isSSL?'p':'a'}">${g.probeType}</span></div>
        <div class="pgn">${g.probeRationale}</div>
      </div>
      <div class="pgc">
        <div class="pgl">Layer selection</div>
        <div style="margin-bottom:4px"><span class="pill ${g.isBirdSpecies&&!g.isSSL?'a':'g'}">${g.layerRec}</span></div>
        <div class="pgn">${g.layerRationale}</div>
      </div>
      <div class="pgc">
        <div class="pgl">Layer importance map</div>
        ${renderLayerDiagram(g.layers)}
        <div class="pgn" style="margin-top:4px">${g.costNote}</div>
      </div>
    </div>
  </div>`;
}

function renderModel(id,rank,leaf){
  const m=MODELS[id];if(!m)return'';
  const isTop=rank===0;
  const rankLabels=['1','2','3'],rankCls=['r1','r2','r3'];
  let scoreHtml='<div class="sg">';
  for(const[k,v]of Object.entries(m.scores)){
    const pct=Math.round((v/maxScores[k])*100);
    scoreHtml+=`<div><div class="sl2">${k}</div><div class="sbw"><div class="sb"><div class="sf ${scoreColors[k]}" style="width:${pct}%"></div></div><div class="sv">${v.toFixed(1)}</div></div></div>`;
  }
  scoreHtml+='</div>';
  const tags=m.tags.map(t=>`<span class="tag ${t.c}">${t.l}</span>`).join('');
  return`<div class="mc ${isTop?'tp':''}">
    ${isTop?'<div class="tpbanner">Recommended</div>':''}
    <div class="mch">
      <div class="mr ${rankCls[rank]}">${rankLabels[rank]}</div>
      <div class="mnw"><div class="mn">${m.name}</div><div class="mid">${m.id}</div></div>
      <a class="ml" href="${m.hf}" target="_blank">View on HuggingFace ↗</a>
    </div>
    <div class="mt">${tags}</div>
    <div class="md">${m.desc}</div>
    ${scoreHtml}
    ${isTop&&leaf?renderProbeGuidance(id,leaf):''}
    <code class="lc">from avex import load_model\nmodel = load_model("${m.id}", device="cuda")</code>
  </div>`;
}

function renderResults(ids,leaf){
  const top3=ids.slice(0,3);
  let html=`<div class="rh"><div class="rt">Recommended models</div><div class="rs">Models ranked by suitability for your use case. Probing strategy guidance is drawn from our <a href="https://arxiv.org/abs/2508.11845" target="_blank" style="color:var(--eg)">evaluation across 26 bioacoustics datasets</a> and our <a href="https://arxiv.org/abs/2509.04166" target="_blank" style="color:var(--eg)">probing study</a>.</div></div>`;
  html+=top3.map((id,i)=>renderModel(id,i,leaf)).join('');
  if(ids.length>1){
    html+=`<button class="ct2" style="margin-top:1.5rem" onclick="toggleCompare()">${state.showCompare?'▲ Hide comparison table':'▼ Compare all metrics side by side'}</button>`;
    if(state.showCompare){
      const keys=Object.keys(MODELS[top3[0]].scores);
      html+=`<div style="overflow-x:auto"><table class="ctb"><thead><tr><th>Model</th>${keys.map(k=>`<th>${k}</th>`).join('')}<th>Arch</th><th>Training</th></tr></thead><tbody>`;
      for(const id of top3){
        const m=MODELS[id];
        const vals=keys.map(k=>{const v=m.scores[k];const cls=v/maxScores[k]>.85?'vg':v/maxScores[k]>.7?'vm':'';return`<td class="${cls}">${v.toFixed(1)}</td>`;}).join('');
        html+=`<tr><td>${m.name}</td>${vals}<td>${m.arch}</td><td>${m.training}</td></tr>`;
      }
      html+='</tbody></table></div>';
    }
  }
  html+=renderNavRow(true);
  return html;
}

function render(){
  const leaf=getLeaf();
  const node=leaf?null:getCurrentNode();
  const atStart=state.path.length===0;
  const intro=document.getElementById('ms-intro');
  if(intro)intro.style.display=atStart?'':'none';
  let html=renderCrumbs();
  if(leaf)html+=renderResults(leaf.result,leaf);
  else if(node)html+=renderQuestion(node);
  document.getElementById('app').innerHTML=html;
}

function selectOption(id,label){state.path.push(id);state.selections.push(label);state.showCompare=true;render();}
function goBack(){state.path.pop();state.selections.pop();state.showCompare=true;render();}
function goTo(idx){state.path=state.path.slice(0,idx);state.selections=state.selections.slice(0,idx);state.showCompare=true;render();}
function resetAll(){state.path=[];state.selections=[];state.showCompare=true;render();}
function toggleCompare(){state.showCompare=!state.showCompare;render();}
render();
</script>

```
