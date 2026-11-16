import React, {useState} from 'react';
import {DataTable,EvidenceChart,ReportSection,RichNarrative,useDataApp} from '../../data-app-public.jsx';
import './study.css';
const fmt=n=>n==null?'n/a':Number(n).toExponential(2);
const cols=[{key:'rank',label:'Rank'},{key:'method',label:'Method'},{key:'community',label:'Cluster'},{key:'proteins',label:'Size'},{key:'protein_set_hits',label:'MoonLitDB'},{key:'fold_enrichment',label:'Fold',renderCell:v=>Number(v).toFixed(2)+'×'},{key:'qvalue',label:'Global q',renderCell:fmt},{key:'significance',label:'Stars'},{key:'label',label:'Leading functional label'}];
export function ReportContent(){
 const {snapshot,appTitle}=useDataApp();
 const summary=snapshot.queries.summary.rows,clusters=snapshot.queries.clusters.rows;
 const [network,setNetwork]=useState('BioPlex'),[selected,setSelected]=useState(null);
 const rows=clusters.filter(r=>r.network===network),detail=rows.find(r=>r.rank===selected)||rows[0];
 const total=summary.reduce((n,r)=>n+r.selected,0);
 const overview=summary.map(r=>`**${r.network}:** ${r.selected} selected results from ${r.methods} methods, representing ${r.unique_memberships} distinct memberships. The top cluster contains ${r.top_hits}/${r.top_size} MoonLitDB proteins (${r.top_fold.toFixed(2)}× enrichment), with leading label **${r.top_label}**.`).join('\n\n');
 const bpThemes=snapshot.queries.themes.rows.filter(r=>r.network==='BioPlex');
 return <article className="report-content study-report">
 <header className="report-hero"><h1>{appTitle}</h1><RichNarrative id="study-deck" value="Small communities enriched for the MoonLitDB protein set, with GO, Reactome and Swiss-Prot FUNCTION auto-annotation."/></header>
 <ReportSection id="study-summary" queryId="summary" title="Executive summary" showHeading={false}>
 <RichNarrative id="study-summary-text" value={`## Executive summary

**${total} enriched cluster results qualify with fewer than 200 proteins and global BH q < 0.05.**

${overview}

Overlapping and repeated clusters are not independent discoveries. Exact membership deduplication removes identical protein groups only.`}/></ReportSection>
 <ReportSection id="study-populations" queryId="summary" title="Study populations">
 <DataTable rows={summary} columns={[{key:'network',label:'Interactome'},{key:'background',label:'Network proteins'},{key:'moonlitdb',label:'MoonLitDB present'},{key:'compatible_runs',label:'Saved runs'},{key:'tested',label:'Communities tested'},{key:'selected',label:'Selected'},{key:'unique_memberships',label:'Distinct memberships'}]} searchable={false} caption="Study populations by network"/>
 </ReportSection>
 <EvidenceChart id="study-selection-chart" queryId="summary" title="Enriched clusters with fewer than 200 proteins" rows={summary} sourceRows={summary} height={300} spec={{type:'bar',x:'network',y:'selected',yLabel:'Selected cluster results',showValues:true}}/>
 <RichNarrative id="selection-chart-note" value="Counts are descriptive; no hypothesis test compares networks. They depend on the saved methods, partitions and network backgrounds, and do not measure which interactome is better."/>
 <ReportSection id="study-bioplex" queryId="themes" title="BioPlex repeatedly recovers aminoacylation-related communities" showHeading={false}>
 <RichNarrative id="study-bioplex-text" value={`## BioPlex functional themes

Leading GO biological-process labels summarize the distinct selected protein memberships. The chart shows their frequencies after exact membership deduplication. Related terms and overlapping groups can repeat the same biological signal; these are descriptive frequencies, not independent pathway discoveries.`}/></ReportSection>
 <EvidenceChart id="study-bioplex-themes" queryId="themes" title="Leading GO BP labels across distinct BioPlex memberships" rows={bpThemes} sourceRows={snapshot.queries.themes.rows} height={450} spec={{type:'horizontalBar',x:'term',y:'clusters',showValues:true,yLabel:'Distinct protein memberships'}}/>
 <RichNarrative id="themes-note" value="Label counts are descriptive, with no significance test. Individual functional enrichment tests are available in the cluster details."/>
 <ReportSection id="study-tagc" queryId="clusters" title="TAGC functional annotations" showHeading={false}>
 <RichNarrative id="study-tagc-text" value={`## TAGC functional annotations

${overview.split("\n\n")[1]}

Explore all functional terms and memberships in the catalogue. A leading label is a summary rather than a complete description; absence of significant annotation is not evidence of missing function.`}/></ReportSection>
 <ReportSection id="study-catalogue" queryId="clusters" title="Explore every selected cluster">
 <label>Interactome <select value={network} onChange={e=>{setNetwork(e.target.value);setSelected(null)}}><option>BioPlex</option><option>TAGC_binary2025</option></select></label>
 <RichNarrative id="catalogue-help" value="Select a row for its functional terms and complete UniProt membership. Search by method, cluster or leading label. Ranking uses global community q, then fold enrichment and MoonLitDB hits."/>
 <DataTable rows={rows} columns={cols} rowKey="rank" selectedRowKey={detail?.rank} onRowSelect={r=>setSelected(r.rank)} rowActionLabel={r=>'Inspect cluster '+r.community} caption={'Ranked MoonLitDB-enriched communities in '+network}/>
 {detail&&<section className="study-detail" data-reviewed-rows>
 <h3>{network} · rank {detail.rank} · {detail.method} · cluster {detail.community}</h3>
 <p>{detail.protein_set_hits}/{detail.proteins} MoonLitDB proteins · {detail.fold_enrichment.toFixed(2)}× enrichment · global q {fmt(detail.qvalue)} {detail.significance}</p>
 <p><strong>GO BP:</strong> {detail.GO_BP_enrichment||'No significant label'}<br/><strong>GO MF:</strong> {detail.GO_MF_enrichment||'No significant label'}<br/><strong>GO CC:</strong> {detail.GO_CC_enrichment||'No significant label'}<br/><strong>Reactome:</strong> {detail.Reactome_enrichment||'No significant label'}</p>
 <p><strong>Swiss-Prot FUNCTION keywords:</strong> {detail.UniProt_FUNCTION_tfidf||'Unavailable'}. Keywords are descriptive.</p>
 <details><summary>All {detail.members.length} protein accessions</summary><p className="study-members">{detail.members.join(' · ')}</p></details>
 <DataTable rows={detail.terms} columns={[{key:'library',label:'Library'},{key:'term_id',label:'Term ID'},{key:'term_name',label:'Functional term'},{key:'query_hits',label:'Hits'},{key:'query_size',label:'Size'},{key:'fold_enrichment',label:'Fold',renderCell:v=>Number(v).toFixed(2)},{key:'qvalue',label:'Annotation q',renderCell:fmt},{key:'significance',label:'Stars'}]} caption="Significant functional terms for the full cluster"/>
 </section>}
 </ReportSection>
 <ReportSection id="study-methods" queryId="summary" title="Methods and interpretation" showHeading={false}>
 <RichNarrative id="study-methods-text" value={'## Methods and interpretation\n\n**Selection.** All 41 successful, fingerprint-compatible saved runs in each network were analyzed without rerunning clustering. One-sided hypergeometric tests compare MoonLitDB membership with the full corresponding interactome. Community BH correction includes all **111,858 BioPlex** or **105,041 TAGC** communities before filtering. Selection requires global q < 0.05 and strictly fewer than 200 proteins, capped at 200 per network. No saved runs were excluded.\n\n**Auto-annotation.** GO BP, MF and CC use local human GO annotations with ancestor propagation; Reactome uses the local UniProt mapping. Enrichment uses the full network background. The existing workflow applies BH across all eligible terms, libraries and nonempty full/core/interface subsets of selected clusters **within each saved run**. Annotation q-values and MoonLitDB community q-values answer different questions and have different correction families. The catalogue shows significant full-cluster terms; annotation bundles retain complete results.\n\n**Community stars:** * q < 0.05; ** q < 0.01; *** q < 0.001; **** q < 0.0001. ns means a valid nonsignificant test; n/a means unavailable. Only significant communities are selected.\n\n**Limitations.** Network coverage, ascertainment, overlapping partitions and correlated annotations affect interpretation. Exact membership deduplication removes identical sets only; remaining groups can overlap heavily. Leading GO labels are summaries, not exclusive functions or evidence of novel moonlighting. Functional enrichment after selection is exploratory. No literature or experimental validation was performed.'}/>
 </ReportSection>
 </article>;
}
