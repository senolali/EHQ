"""
PCQ (Post-Cutoff Event Questions) Generation Module
====================================================
EHQ-3000 genisletmesi: Haziran-Temmuz 2026 gercek olay penceresi.

TASARIM:
  - Olay penceresi: Haziran-Temmuz 2026
  - Tum 20 test modelinin cutoff'u en gec Ocak 2026 -> k_i=0 %100 garantili
  - Gold answer: gercek kaynaktan (hakem paneline GEREK YOK)
  - Uretici: Mistral Large (ASU) - kaynaktan soru + gold answer uretir
  - Dokumansiz soru: model kendi egitim bilgisinden cevaplamayi deniyor
    ama Haz-Tem 2026'yi bilmiyor -> abstain etmeli (EHQ ozu)

KAYNAK HAVUZU (web'den dogrulanmis):
  PCQ-SPO: 2026 FIFA Dunya Kupasi (Haz 11 - Tem 19, 2026)
  PCQ-POL: 2026 NATO Zirvesi (Tem 7-8, Ankara)
  PCQ-SCI: Haziran-Temmuz 2026 AI/teknoloji olaylari
  PCQ-ECO: IMF WEO Guncelleme + ekonomi olaylari (Temmuz 2026)
  PCQ-WOR: Dunya olaylari (Venezuela depremi, UK PM degisimi vb.)

Ortam degiskeni: ASU_CREATEAI_TOKEN
"""

import os
import re
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

from asu_client import asu_query

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pcq_generator")

# ----------------------------------------------------------------------
# Yapilandirma
# ----------------------------------------------------------------------

EVENT_WINDOW = "June-July 2026"

GENERATOR_MODEL    = "mistral-large"
GENERATOR_PROVIDER = "aws"

# Gercek olay kaynak havuzu - web'den dogrulanmis
SOURCE_FACTS = {
"PCQ-SPO": """
2026 FIFA World Cup (June 11 - July 19, 2026, hosted by USA/Canada/Mexico):
- CHAMPION: Spain defeated Argentina 1-0 after extra time in the final on July 19, 2026
- Winning goal: Ferran Torres (substitute), 106th minute
- Final venue: New York New Jersey Stadium (MetLife Stadium), East Rutherford, New Jersey
- Argentina's Enzo Fernandez was sent off (red card, two yellows) in second half
- Third place: England defeated France 6-4 (Bukayo Saka hat-trick), July 18, Hard Rock Stadium, Miami
- Golden Boot: Kylian Mbappe (France), 10 goals in tournament, becoming the tournament's all-time leading scorer; Messi runner-up with 8 goals
- Golden Glove: Emiliano Martinez (Argentina)
- Host nations eliminated in Round of 16: Canada (lost to Morocco), Mexico (lost to England), USA (lost to Belgium)
- Tournament format: 48 teams, 12 groups of 4
- Group A: Mexico 1st (beat South Africa 2-0, South Korea 1-0, Czechia 3-0)
- Group B: Canada 1st (beat Qatar 6-0 in match 2)
- Group D: USA 1st (beat Paraguay 4-1 in opener, June 12)
- Germany beat Curacao 7-1 (Group E, June 14)
- Lionel Messi became first player to make 30 World Cup appearances
- Semifinalists: Spain, Argentina, England, France
- Spain's second World Cup title (first was 2010)
- World Cup final: Spain did not allow Argentina a single shot on goal in the entire match
- With Argentina's final loss, no country has won back-to-back World Cup titles since Brazil (1958, 1962)
- Wimbledon 2026 Men's Singles champion: Jannik Sinner, defeated Alexander Zverev 6-7,7-6,6-3,6-4 (back-to-back title, 5th major)
- Wimbledon 2026 Women's Singles champion: Linda Noskova, defeated Karolina Muchova 6-2,5-7,6-3 (July 11, 2026)
- French Open 2026 Men's Singles champion: Alexander Zverev, defeated Flavio Cobolli (June 7, 2026), his first Grand Slam title
- French Open 2026 Women's Singles champion: Mirra Andreeva (19), defeated Maja Chwalinska 6-3,6-3
- 2026 NBA Finals champion: New York Knicks defeated San Antonio Spurs 4-1, first Knicks title since 1973
- 2026 NBA Finals MVP: Jalen Brunson (scored 45 points in Game 5)
- 2026 NBA Eastern Conference Finals: Knicks swept the Cleveland Cavaliers 4-0
- 2026 NBA Western Conference Finals: Spurs beat the Oklahoma City Thunder 4-3
- 2026 Stanley Cup champion: Carolina Hurricanes defeated Vegas Golden Knights (3-0 in Game 6), first Cup since 2006
- 2026 Stanley Cup playoff MVP: Jordan Staal
- 2026 Tour de France overall winner: Tadej Pogacar (5th title, 3rd consecutive), ~6.5 minutes ahead of Remco Evenepoel; race ended July 26, 2026
- 2026 Tour de France final stage (Stage 21) winner: Mathieu van der Poel (sprint finish)
- 2026 Tour de France podium: Remco Evenepoel 2nd, Isaac del Toro 3rd (Tour debut)
- 2026 Tour de France final stage route was shortened because police/security resources were reassigned to fight wildfires
- 24 Hours of Le Mans 2026 winner: Toyota's #7 GR010 Hybrid (Mike Conway, Kamui Kobayashi, Nyck de Vries), by 11 seconds over the #20 BMW M Hybrid V8
- 2026 US Open (golf) winner: Wyndham Clark, won by one stroke at Shinnecock Hills (June 18-21, 2026), a wire-to-wire victory
- 2026 Open Championship (golf) winner: Ryan Fox at Royal Birkdale, 10-under, one shot clear of Cameron Young (July 16-19, 2026), his maiden major
- 2026 F1 Spanish GP winner: Lewis Hamilton, his first race win for Ferrari (106th career win), June 15, 2026
- 2026 F1 Austrian GP winner: George Russell, from pole, 1.6s ahead of Max Verstappen (June 26-28, 2026)
- 2026 F1 British GP winner: Charles Leclerc, ahead of Russell and Hamilton, his first win at Silverstone (July 5, 2026)
- 2026 F1 Belgian GP winner: Kimi Antonelli, his 6th win of the season, ahead of Leclerc and Verstappen (July 19, 2026)
- 2026 F1 Hungarian GP winner: Lando Norris, ahead of Verstappen and Antonelli by ~15 seconds (July 26, 2026)
- 2026 MLB All-Star Game: American League beat National League 4-0 (July 14, 2026, Citizens Bank Park, Philadelphia); MVP Cody Bellinger (New York Yankees)
- New Zealand won a 3-match Test cricket series in England 2-1, despite losing the 1st Test at Lord's (June 4-7) by 115 runs
- Ireland swept India 2-0 in a two-match T20I series in Belfast (June 26 and June 28, 2026), India's first-ever bilateral T20I series loss to Ireland
- Bangladesh won a 3-match ODI cricket series against Australia 2-1 (June 9-14, 2026, Dhaka/Chattogram)
- The 2026 Major League Cricket (US) season ran June 18 - July 18, 2026
- UFC 329 "McGregor vs. Holloway 2" was held July 11, 2026 at T-Mobile Arena, Paradise, Nevada
- UFC Fight Night: Fiziev vs. Torres was held June 27, 2026 in Baku, Azerbaijan
- Jesse "Bam" Rodriguez defeated Antonio Vargas by 6th-round KO on June 13, 2026 in Glendale, Arizona to win the WBA bantamweight title, becoming a three-division world champion
- At the Rome Golden Gala (June 4, 2026) Diamond League meet, Noah Lyles won the men's 100m in 9.88 seconds
- At the Paris Diamond League (June 28, 2026), Armand Duplantis cleared a meet-record 6.13m in the pole vault
- At the Paris Diamond League (June 28, 2026), Busang Collen Kebinatshipi won the 400m in a meeting-record 43.54
- At the Monaco Diamond League (July 10, 2026), Miltiadis Tentoglou produced a world-leading 8.61m long jump
- At the Prefontaine Classic, Eugene (July 3-4, 2026), Melissa Jefferson-Wooden won the women's 100m in 10.78, ahead of Sha'Carri Richardson
- Donna Vekic defeated Emma Raducanu 6-0, 7-6 in the 2026 Queen's Club Championships women's singles final (June 8-14)
- Frances Tiafoe defeated Taylor Fritz 6-4, 6-4 in the 2026 Halle Open final, his 4th ATP title
- Nelly Korda won the 2026 US Women's Open golf title by one stroke over Charley Hull and Gaby Lopez at Riviera Country Club (June 4-7, 2026), her 4th career major
- Haeran Ryu won the 2026 KPMG Women's PGA Championship at Hazeltine National, her first major title
- Haeran Ryu won the 2026 Evian Championship (July 9-12, 2026), her second consecutive major of the year, beating Brooke Henderson in a playoff
- The 2026 NBA Draft was held June 23-24, 2026 at Barclays Center, Brooklyn; the Washington Wizards used the No. 1 pick on AJ Dybantsa
- The New York Liberty defeated the Las Vegas Aces 93-85 to win the 2026 WNBA Commissioner's Cup (June 30, 2026), becoming the first two-time champions
- The 2026 WNBA All-Star Game was scheduled for July 25, 2026 at the United Center in Chicago
- Leviatan defeated Paper Rex 3-2 to win Valorant Masters London 2026 (June 21, 2026), their first international VCT title
- The 2026 Commonwealth Games opened in Glasgow on July 23, 2026 with an OVO Hydro ceremony
- Corey Heim won the 2026 Brickyard 400 NASCAR Cup race at Indianapolis on July 26, 2026, his second career Cup Series win
- The 2026 FIFA World Cup drew a record total attendance of 6,810,966 fans across the tournament, nearly double the 1994 USA record
- FIFA recorded its highest-ever single-day World Cup attendance on June 25, 2026, with 426,834 spectators
- At the 2026 World Cup quarterfinals, France beat Morocco 2-0, a rematch of the 2022 semifinal
- At the 2026 World Cup semifinals, Spain beat France 2-0 in Dallas (July 14, 2026) and Argentina beat England 2-1 in Atlanta (July 15, 2026)
- The Netherlands beat Sweden 5-1 in 2026 World Cup group play (June 20, 2026)
- Turkiye beat the USA 3-2 in an upset result in the final 2026 World Cup group match (June 25, 2026)
- The 2026 World Cup opening match was Mexico 2-0 South Africa at Estadio Azteca (June 11, 2026); goals by Julian Quinones and Raul Jimenez
- 2026 World Cup Round of 32: Belgium beat Senegal 3-2 after extra time
- 2026 World Cup Round of 32: Morocco eliminated the Netherlands 3-2 on penalties after a 1-1 draw
- 2026 World Cup Round of 16: Belgium beat the USA 4-1
- 2026 World Cup Round of 16: Morocco beat Canada 3-0
- 2026 World Cup quarterfinal: Spain beat Belgium 2-1
- 2026 World Cup quarterfinal: England beat Norway 2-1 after extra time
- 2026 World Cup quarterfinal: Argentina came from behind to beat Switzerland 3-1 after extra time
- 2026 World Cup semifinal (Dallas, July 14): Spain beat France 2-0, goals by Mikel Oyarzabal (penalty) and Pedro Porro
- 2026 World Cup semifinal (July 15): Argentina beat England 2-1, with Messi assisting both Argentina goals
- In the 2026 World Cup final, Argentina keeper Emiliano Martinez set a World Cup final record with 11 saves
- Cristiano Ronaldo became the oldest player to score a multi-goal haul in a World Cup match (41 years, 138 days) with a brace in Portugal's 5-0 win over Uzbekistan
- Lamine Yamal became the youngest Spanish player ever to score at a World Cup, scoring vs. Saudi Arabia at 18 years, 343 days
- MetLife Stadium hosted 7 matches at the 2026 World Cup for a tournament-high 564,523 total spectators
- Madison Keys defeated Tatjana Maria 7-5, 6-4 to win the 2026 Eastbourne Open, her third Eastbourne title
- Zizou Bergs defeated Ugo Humbert to win the 2026 Eastbourne Open men's title, his first ATP Tour title
- Alejandro Davidovich Fokina defeated Ethan Quinn to win the 2026 Mallorca Championships, his first ATP singles title
- Karolina Muchova beat Naomi Osaka (retired injured) to win the 2026 Bad Homburg Open, her first grass-court title
- Tom Kim won the 2026 Genesis Scottish Open at The Renaissance Club, North Berwick
- Chris Gotterup won the 2026 John Deere Classic at TPC Deere Run, his third PGA Tour win of 2026
- Viktor Hovland beat Scottie Scheffler in a playoff to win the 2026 Travelers Championship, the first Norwegian to win the event
- Cooper Lutkenhaus won the men's 800m at the Stockholm Diamond League (June 7, 2026) in 1:42.70
- Ai Ogura won the MotoGP Dutch TT at Assen (June 28, 2026), his first MotoGP career win
- Alex Palou won the IndyCar Chevrolet Detroit Grand Prix 2026, his 4th win of the season
- New South Wales beat Queensland 30-12 in NRL State of Origin Game 3 (July 8, 2026) to win the series 2-1; Nathan Cleary won Player of the Series
- Luke Littler defeated Gerwyn Price 18-9 in the final of the 2026 World Matchplay darts, retaining his title
- Sawyer Lindblad won the World Surf League's VIVO Rio Pro, concluding June 26, 2026
""",

"PCQ-POL": """
2026 NATO Summit - Ankara, Turkey (July 7-8, 2026):
- Host: Turkey (Turkiye) - second time hosting, first was Istanbul 2004
- Venue: Besiktas Presidential Complex (Kulliye), Ankara
- This was the 36th NATO Summit
- Secretary General: Mark Rutte
- NATO has 32 member states
- Key priorities: defense investment (5% GDP target), Ukraine support, transatlantic industrial cooperation
- NATO Defence Industry Forum (NSDIF26) held July 7, 2026, at ATO Congresium, Ankara
- Allied leaders reaffirmed Article 5 commitment
- European allies and Canada collectively spent additional $1.2 trillion on defense over past decade
- Summit Declaration published: "Ankara Summit Declaration"
- Follows 2025 The Hague Summit; precedes 2027 Albania Summit
- US President Trump's NATO spending demands addressed at summit
- Rutte's goal: "NATO 3.0 - stronger Europe in stronger NATO"
- UK: John Healey resigned as Defence Secretary before summit over funding disputes
- At the NATO summit, Zelenskyy and Trump held a bilateral meeting at 14:30 local time, July 8, 2026
- At that meeting, Trump told Zelenskyy the US would license Ukraine to produce Patriot air-defense missiles domestically
- Rutte convened a NATO Indo-Pacific-partner (IP4) side-meeting in Ankara with South Korea, Japan, Australia, New Zealand representatives
- UK PM Keir Starmer announced his resignation on June 22, 2026, following heavy Reform UK gains and cabinet resignations
- Labour MP Josh Simons resigned his Makerfield seat to let Andy Burnham stand for it
- Andy Burnham won the June 18, 2026 Makerfield by-election with 54.8% vs. Reform UK's 34.5%
- UK Labour leadership contest: nominations opened July 9, closed July 16, 2026
- Andy Burnham was nominated by 379 Labour MPs (over 94% of the parliamentary party)
- Andy Burnham was declared Labour leader on July 17, 2026
- Andy Burnham became UK Prime Minister on July 20, 2026, three days after being declared Labour leader
- New UK PM Andy Burnham named John Healey as Chancellor of the Exchequer (July 20, 2026)
- US Supreme Court ruled 6-3 in Trump v. Slaughter (June 29, 2026), overruling the 1935 Humphrey's Executor precedent on independent-agency removal protections
- The same day, the Court preserved limited removal protections for Federal Reserve governors in the companion case Trump v. Cook
- US Supreme Court ruled 6-3 in NRSC v. FEC (June 30, 2026), striking down federal limits on coordinated party spending with candidates
- Ethiopia held its general election June 1, 2026; the ruling Prosperity Party won 438 parliamentary seats
- South Korea held nationwide local elections on June 3, 2026
- South Korean President Lee Jae-myung marked his first anniversary in office on June 4, 2026
- India's Uttarakhand state held municipal elections June 9, 2026, results declared June 11, 2026
- 2026 G7 Summit was held June 15-17, 2026 in Evian-les-Bains, France
- European Council summit held in Brussels June 18-19, 2026 (Ukraine, Middle East, defense, migration on agenda)
- EU-Western Balkans Summit held in Tivat, Montenegro, on June 5, 2026
- Ukraine's EU accession: the Intergovernmental Conference opened the "fundamentals" negotiating cluster on June 15, 2026
- Moldova held a second EU accession conference, also opening its fundamentals cluster, on June 15, 2026
- Colombia held its presidential runoff June 21, 2026: Abelardo de la Espriella won with 49.66% vs. Ivan Cepeda's 48.70% (narrowest runoff margin in Colombian history)
- Ivan Cepeda formally conceded defeat in Colombia's election on June 24, 2026
- Israel's Knesset passed the first reading of a bill to dissolve parliament by a 106-0 vote in early June 2026
- The Knesset formally dissolved itself on July 17, 2026 via a 62-0 vote, setting an election for October 27, 2026
- Taiwan President Lai Ching-te declared "Of course Taiwan is a country" at the start of a national tour, June 22, 2026
- Taiwan held tabletop exercises June 25, 2026 simulating a response to a potential PRC maritime "quarantine"
- German Health Minister Nina Warken replaced Thorsten Frei as Head of the Federal Chancellery, becoming the first woman to hold that office
- Thorsten Frei took over as CDU/CSU Bundestag group leader, replacing Jens Spahn
- CDU Secretary-General Carsten Linnemann became Germany's new Health Minister
- German Chancellor Merz completed his cabinet reshuffle by naming a new transport minister on July 27, 2026
- Iraqi PM Ali al-Zaidi met President Trump at the White House on July 14, 2026
- At that meeting, Trump and al-Zaidi announced remaining US forces would fully withdraw from Iraq by September 30, 2026
- Australia signed a A$2.5 billion agreement to export its Over-the-Horizon Radar system to Canada (June 22, 2026)
- Poland filed a declaration of intervention in the ICJ case Lithuania v. Belarus on June 30, 2026
- The European Union filed a Memorial in the ICJ case Lithuania v. Belarus on July 20, 2026
- Nicolas Maduro and US DOJ prosecutors proposed a June 2027 trial start date on narco-terrorism charges (July 21, 2026)
- The Philippines hosted ASEAN foreign ministers' meetings in Manila July 20-24, 2026, marking the 50th anniversary of the Treaty of Amity and Cooperation
- Keiko Fujimori won Peru's June 7, 2026 presidential runoff with 50.135% of valid votes vs. Roberto Sanchez's 49.865%
- Keiko Fujimori was sworn in as Peru's president on July 28, 2026 in Lima, becoming Peru's first elected female head of state
- South African President Cyril Ramaphosa reshuffled his cabinet around June 30-July 1, 2026 at the Democratic Alliance's request, affecting six ministries
- On June 12, 2026, Nigerian President Tinubu's Democracy Day address announced recruitment of over 50,000 new police officers
- The 49th Ordinary Session of the African Union Executive Council was held June 24-25, 2026 in El Alamein, Egypt
- New UK PM Andy Burnham was formally appointed by King Charles at Buckingham Palace on July 20, 2026; Rachel Reeves left as Chancellor and David Lammy departed as Deputy PM
- The UK Supreme Court ruled by a 3-2 majority (reported July 27, 2026) that Bahrain cannot claim state immunity in a spyware lawsuit brought by activists Saeed Shehabi and Moosa Mohammed
- On June 2, 2026, the UK Supreme Court overruled its own precedent in P v Cheshire West and Chester Council, redefining "deprivation of liberty" under the Mental Capacity Act 2005
- A referendum was held in Slovakia on July 4, 2026 on cancelling lifelong payments for former PMs/parliament speakers and restoring the Special Prosecutor's Office
- Xi Jinping spoke at the Great Hall of the People in Beijing on July 1, 2026, marking the 105th anniversary of the Chinese Communist Party
- China hosted the World AI Conference and High-Level Meeting on Global AI Governance in Shanghai, July 17-20, 2026, where Xi announced the Global AI Governance Initiative
- On June 3-4, 2026, Kim Jong Un inspected a new North Korean plant producing weapons-grade nuclear material
- Xi Jinping visited Pyongyang June 8-9, 2026, his first North Korea visit in nearly seven years
- The "21st Century Road to Housing Act" became US law without Trump's signature around July 10, 2026, described as the largest housing affordability legislation in decades
- The US House passed the Sunshine Protection Act (permanent daylight saving time) in mid-July 2026 by a vote of 308-117
- The UN Security Council unanimously adopted Resolution 2823 on June 23, 2026, on accountability for crimes against UN peacekeepers, put forward by Denmark and Pakistan
- Japan PM Sanae Takaichi visited Delhi on July 2, 2026, attending the Japan-India Joint Economic Forum with PM Modi
- French President Macron chaired an emergency cabinet meeting on July 27, 2026 over wildfires approaching Bordeaux
- Italy's Chamber of Deputies rejected an amendment to Meloni's electoral reform by one vote (188-187) on July 14, 2026, her government's first parliamentary defeat
- Italy's lower house approved Meloni's electoral reform bill 217-152 on July 16, 2026, shifting toward a proportional system
- Spain's Congress of Deputies passed a resolution (177-171) on June 25, 2026 urging PM Pedro Sanchez to resign or call a confidence vote
- Jaroslaw Kaczynski expelled former PM Mateusz Morawiecki and about 30 MPs from Poland's Law and Justice (PiS) party on July 24, 2026
- Hungary's parliament voted 139-6 on July 14, 2026 to remove President Tamas Sulyok from office
- Viktor Orban called for a new era of "resistance politics" for Fidesz on July 25, 2026 following its election defeat
- Kosovo held parliamentary elections June 7, 2026 (its third election in about a year); PM Albin Kurti's Vetevendosje won the most votes but no majority
- Kazakhstan's Constitutional Court ruled July 7, 2026 that President Tokayev may seek another term under the new 2026 Constitution
- A VTsIOM poll released July 10, 2026 showed Russian President Putin's approval falling to 71%
- Philippines VP Sara Duterte's impeachment trial began in early July 2026
- Thailand's Constitutional Court ruled July 9, 2026 that PM Anutin Charnvirakul's 400-billion-baht emergency borrowing decree was constitutional
- Canadian PM Mark Carney called by-elections for August 31, 2026 in three ridings, announced July 26, 2026
- ICC member states voted 82-13 on July 24, 2026 to remove Chief Prosecutor Karim Khan over misconduct findings, the first removal of a sitting ICC chief prosecutor
- The EU and UK signed the Gibraltar Treaty in Brussels on July 14, 2026, resolving Gibraltar's post-Brexit border status
- The US-Iran war resumed in July 2026 after a ceasefire collapsed, with US strikes on Iran for 13+ consecutive nights
- Yemen's Houthis declared a naval blockade on Saudi Arabia in July 2026, opening a new front in the Iran war
- Pope Leo XIV used his Angelus address on July 26, 2026 to appeal for a halt to Mideast attacks
- Pope Leo XIV made a pastoral visit to Lampedusa on July 4, 2026, celebrating Holy Mass there
- Ireland began its six-month EU Council presidency on July 1, 2026
- Irish Taoiseach Micheal Martin visited Kyiv on July 23, 2026 and announced a new 125 million euro support package for Ukraine
""",

"PCQ-SCI": """
AI and Technology Events, June-July 2026:
- GPT-5.5 Instant released by OpenAI (June 2026)
- Google's Gemini 3.5 Flash released (June 2026)
- Anthropic's Claude Opus 4.8 released (June 2026), new performance benchmarks
- GPT-5.6 (Luna, Terra, Sol) released by OpenAI (July 9, 2026) with February 16, 2026 cutoff
- Claude Sonnet 5 released by Anthropic (July 2026), $2/million input tokens, $10/million output
- Kimi K2.7 Code: first open-weight model inside GitHub Copilot
- Reflection AI raised $6.3 billion in compute commitments through 2029
- NVIDIA Cosmos 3 and Intel Xeon 6+ hardware upgrades announced
- DeepSeek announced custom inference silicon design to reduce Nvidia dependence
- Microsoft cut 4,800 jobs (Xbox margin crisis, July 2026)
- Orion-100B trained 100B parameter model for $1.25/hour
- ZoomMate AI assistant launched at $20/user/month
- US Federal agencies hit July 2 deadline from June 2 AI executive order
- Google Imagen 3 Nano and Pro became widely available (June 2026)
- AMD Advancing AI conference scheduled July 22-23, 2026
- Google I/O 2026 announced AI search enhancements
- SpaceX AI, OpenAI, Meta shipped flagship models within 24 hours (July 2026)
- GitHub moved all Copilot plans to usage-based billing on June 1, 2026 (1,500/7,000/20,000 monthly "AI Credits" for Pro/Pro+/Max)
- NVIDIA launched its Vera Rubin platform (Vera CPUs + Rubin GPUs) June 22, 2026 at ISC High Performance in Hamburg
- NVIDIA unveiled the RTX Spark AI "superchip" for thin laptops/desktops June 1, 2026 at Computex
- IBM announced June 2, 2026 it will invest more than $10 billion in quantum computing over five years, targeting a fault-tolerant quantum computer by 2029
- Apple WWDC 2026 (June 8-12, 2026): announced iOS 27, iPadOS 27, macOS 27 "Golden Gate," and a rebuilt Siri co-built with Google's Gemini team
- Microsoft launched "Microsoft Frontier Company" on July 2, 2026, backed by $2.5 billion, led by Rodrigo Kede Lima
- xAI released Grok 4.5 on July 8, 2026, built on a 1.5-trillion-parameter "V9" foundation, priced $2/$6 per million input/output tokens
- OpenAI introduced "GPT-Live," a full-duplex voice model for ChatGPT Voice, on July 8, 2026
- OpenAI launched "ChatGPT Work," an agent for full jobs, alongside GPT-5.6 on July 9, 2026
- Meta Superintelligence Labs released "Muse Spark 1.1" on July 9, 2026, priced $1.25/$4.25 per million input/output tokens
- Meta introduced "Muse Image," its first image-generation model from Meta Superintelligence Labs (July 2026)
- Meta launched "AI Mode" on Facebook on June 15, 2026, synthesizing answers from public posts
- Google's "Dataland" AI arts museum opened June 20, 2026 at The Grand LA, created with artist Refik Anadol
- Google DeepMind announced a research partnership with film studio A24 on June 22, 2026
- NVIDIA released "Nemotron 3 Nano Omni," an open-weight 30B-parameter omni-modal model, in June 2026
- Anthropic released Claude Opus 5 on July 24, 2026, priced $5/$25 per million input/output tokens, becoming the new Claude Max default
- Moonshot AI's Kimi K3 API went live July 16, 2026 (2.8-trillion-parameter sparse MoE model, 1M-token context); open weights released July 26, 2026
- AMD "Advancing AI" event launched EPYC "Venice" (first Zen 6 x86 server CPU, TSMC 2nm) and the Instinct MI455X GPU (CDNA5, 432GB HBM4)
- TSMC confirmed its A16 chip manufacturing node will enter production in the second half of 2026
- TSMC's June 2026 sales rose 67.9% year-over-year, breaking a four-year seasonal decline pattern
- Qualcomm's June 24, 2026 Investor Day projected data-center revenue reaching $15B by FY2029, unveiling the "Dragonfly C1000" CPU with Meta as a customer
- SpaceX's Starship Flight 13 aborted at T-0 on July 16, 2026 (Raptor engine ignition failures); a third attempt succeeded July 24, 2026, deploying operational Starlink V3 satellites
- JWST provided the strongest evidence yet for "black hole stars" on June 10, 2026
- JWST imaged the dawn and dusk terminators of exoplanet WASP-121b separately in June 2026, finding asymmetric temperatures
- A Leiden/Oxford-led team used JWST to observe a complex of six closely packed galaxies with a growing supermassive black hole in the young universe
- Intellia's HAELO phase 3 trial results published in NEJM (June 12, 2026): a single CRISPR-Cas9 IV dose cut hereditary angioedema attacks by 87% vs. placebo
- Nature published research (reported June 25, 2026) showing next-generation genome-editing tools are more precise than earlier CRISPR forms
- ShinyHunters breached Madison Square Garden Entertainment via voice phishing, publishing 45GB of data (~26 million records) on June 16, 2026
- ShinyHunters claimed to have stolen more than 2.2 million Kodak customer/corporate records, with a June 18, 2026 leak deadline
- DHS confirmed July 1, 2026 that hackers breached the Homeland Security Information Network (HSIN) coordinating 2026 FIFA World Cup security
- Robot.com launched its first humanoid robot, "R-noid," at the Automate 2026 trade show (June 22-25, 2026)
- Waymo operated robotaxis in six US World Cup host cities, completing more than 90,000 trips to stadiums/watch parties
- Solar power generated a record 52 TWh of EU electricity in June 2026 (25% of the bloc's monthly generation)
- California's grid operator (CAISO) supplied 23 GW of solar power on July 10, 2026, covering 72% of regional demand
- The US Department of Energy released its finalized Fusion Science and Technology Roadmap on June 9, 2026
- Together AI closed an $800 million Series C funding round at an $8.3 billion valuation
- Quantum Systems (German drone/robotics defense-tech firm) raised $1.2 billion in a Series D round on July 2, 2026
- AIsphere secured $439 million in a Series C round led by Alibaba Group Holding, completed July 14, 2026
- Peregrine Technologies secured $250 million in a Series D round, valuing the company at $6.8 billion
- Cyera raised $600 million in June 2026, reaching a $12 billion valuation
- Fintech company Ramp secured funding in June 2026 at a $44 billion valuation
- NASA's "Swift Boost" mission launched July 3, 2026 from Kwajalein Atoll to grapple and raise the decaying orbit of the Neil Gehrels Swift Observatory
- UC Irvine astronomers announced discovery of exoplanet GJ 3378b on June 30, 2026, roughly twice Earth's size, in its star's habitable zone
- A NASA-led study (published July 16, 2026) revealed that near-Earth object 1998 SH2, long classified as an asteroid, is actually a weakly active "dark comet"
- The FDA approved Lumvoa (veligrotug-vvze) on June 26, 2026, the first thyroid eye disease drug labeled effective across both active and chronic stages
- The FDA approved Tregzi on June 30, 2026, the first regulatory T-cell-based immunotherapy for chronic GVHD-free survival after stem cell transplant
- An FDA advisory committee voted unanimously 9-0 on June 18, 2026 to endorse Moderna's mRNA flu vaccine mFlusiva for adults 50-64
- France notified WHO on June 24, 2026 of a lab-confirmed Ebola (Bundibugyo virus) case in a doctor returning from the DRC
- WHO's mpox external situation report #67 was published June 26, 2026
- Ransomware group WorldLeaks claimed an attack on Tata Electronics, posting 204,341 files (630.4 GB) to its dark-web leak site on June 12, 2026
- Samsung and Broadcom signed a five-year, $200 billion chip-supply agreement, announced July 24-25, 2026 at an AI summit in San Francisco
- Intel announced layoffs on July 21, 2026 cutting roughly 4,000 US Data Center group positions, including 2,392 in Oregon
- ASML reported Q2 2026 results on July 15, 2026: EPS of $8.82, beating estimates, with net sales of 9.3 billion euros
- Micron reported record Q3 FY2026 results on June 24, 2026: revenue of $41.46 billion, up 346% year-over-year
- China's Jiangmen Underground Neutrino Observatory (JUNO) published its first physics result as a Nature cover article on June 10-11, 2026
- Mistral AI released "Robostral Navigate," a robotics navigation model, reported July 8, 2026
- Alibaba announced Qwen3.8-Max on July 19, 2026 at the World AI Conference in Shanghai, a 2.4-trillion-parameter multimodal model
- Midjourney made V8.1 the default model for all users on June 11, 2026, replacing V7
- Amazon's Zoox unveiled its redesigned "production intent" robotaxi on June 24, 2026, ahead of expansion to Austin and Miami
- Tesla's FSD v14 Lite for Hardware 3 vehicles began wide release on July 21, 2026
- China's securities regulator approved Unitree Robotics' Shanghai STAR Market IPO registration around July 1-3, 2026, valuing the company at roughly $6 billion
- China's "Implementation Opinions on AI Agents," the world's first dedicated AI-agent regulatory category, became enforceable July 15, 2026
- June 2026 was Earth's second-warmest June on record (1.09C above the 20th-century average), the 50th consecutive above-average June
- The LHCb Collaboration announced observation of the Omega-cc+ baryon, completing the family of doubly charmed baryons first predicted over 50 years ago (June 2026)
- An Aalto University-led team used AI to discover two new kagome superconductors, YRu3B2 and LuRu3B2, published June 17, 2026
- Vattenfall selected Rolls-Royce SMR on June 15, 2026 to build three small modular reactors in Sweden, its first new nuclear build in over 40 years
- Anthropic's Claude Fable 5 and Mythos 5 models were suspended under a US Commerce Department export-control order around June 12, 2026
- OpenAI disclosed on July 21-22, 2026 that experimental models breached Hugging Face's production infrastructure after breaking out of a test environment
- President Trump signed an executive order "Promoting Advanced Artificial Intelligence Innovation and Security" on June 2, 2026
- Microsoft's July 2026 Patch Tuesday (July 14, 2026) fixed a record 570 vulnerabilities, including two actively-exploited zero-days
- Citrix disclosed CVE-2026-8451, a NetScaler ADC/Gateway vulnerability, on June 30, 2026; it was exploited in the wild within 24 hours
- DJI filed a patent-infringement lawsuit against Insta360 around June 11, 2026 over its Ultra product line
- "Halo: Campaign Evolved" launched in early access July 23, 2026, with full release July 28, 2026
- The Esports World Cup 2026 began in Paris, France on July 6, 2026, with a prize pool exceeding $75 million
- Wear OS 7 began rolling out June 16, 2026, starting with Pixel Watch 2, 3, and 4
""",

"PCQ-ECO": """
Economic Events, June-July 2026:
- IMF World Economic Outlook Update published July 8, 2026
- IMF assumed Strait of Hormuz begins reopening mid-July, normalizing by March 2027
- Average oil price assumption: $89/barrel for 2026 (based on June 10 market pricing)
- US-Iran conflict affecting Strait of Hormuz (approximately 20% of world oil passes through)
- IMF projected global growth with downside risks due to geopolitical uncertainty
- Iran announced plans to "completely block" Strait of Hormuz (June 2026)
- US and Iran exchanged strikes targeting infrastructure and military targets (July 18, 2026)
- Venezuela earthquakes (magnitude 7.2 and 7.5, June 24, 2026) caused economic disruption
- 1,430 confirmed dead in Venezuela earthquake, thousands missing
- UK Prime Minister change: Keir Starmer ousted, Andy Burnham became new PM (July 20, 2026)
- Reflection AI secured $6.3 billion compute deal through 2029
- AI hiring: IBM reported 8% of IT roles now "AI-focused" as of June 2026
- Taylor Swift and Travis Kelce married July 3, 2026 at Madison Square Garden (1,000 guests)
- US celebrated 250th anniversary (semiquincentennial) on July 4, 2026
- The US Federal Reserve held its rate at 3.50%-3.75% on June 17, 2026, Kevin Warsh's first meeting as Fed Chair
- The ECB raised its deposit rate to 2.25% on June 11, 2026, its first hike since 2023
- The Bank of Japan raised its policy rate to 1.00% on June 16, 2026, the highest since 1995
- Turkey's central bank held its policy rate at 37% on June 11, 2026 and again on July 23, 2026
- India's RBI held its repo rate at 5.25% in its June 2026 meeting
- US nonfarm payrolls rose just 57,000 in June 2026 (reported July 2, 2026), missing forecasts
- US unemployment rate fell to 4.2% in June 2026
- US June 2026 CPI fell 0.4% month-over-month, the biggest monthly decline in 6+ years; annual rate fell to 3.5%
- China's Q2 2026 GDP grew 4.3% year-on-year, the slowest quarterly growth since end-2022
- China's Q2 2026 exports surged 27% year-on-year
- Eurozone flash inflation for June 2026 came in at 2.8% year-on-year
- 25 tracked companies announced 13,532 job cuts in July 2026; Microsoft's cut of 4,664 positions was the largest single cut
- 2026 YTD US layoffs totaled 322 events affecting 205,832 workers by July 28, 2026; Oracle's 30,000-job cut was the largest single event
- The US IPO market raised a record $97.9 billion across 19 IPOs in June 2026
- SpaceX IPO'd on Nasdaq (ticker SPCX) on June 12, 2026, priced at $135/share, raising $85.7 billion, the largest IPO debut ever
- SpaceX's share price hit $225 within days of its IPO, pushing its market cap above $3 trillion
- Quantinuum Inc. IPO'd on Nasdaq June 4, 2026, raising about $1.68 billion priced at $60/share
- The S&P 500 closed at a record 7,537.43 on July 6, 2026
- The Dow Jones Industrial Average hit a record close of 53,055.91 on July 6, 2026
- Bitcoin climbed back above $63,000 on July 4, 2026
- Trump and Iranian President Masoud Pezeshkian signed a memorandum of understanding to end the war on June 17, 2026
- Brent crude jumped over 9% on July 12, 2026, its biggest daily gain since 2020
- OPEC+ agreed June 7, 2026 to raise oil production by 188,000 barrels per day for July 2026
- Netflix's Q2 2026 revenue was $12.56 billion, up 13% year-over-year
- TSMC's Q2 2026 revenue was NT$1,270.38 billion, up 36.0% year-over-year
- Tesla's Q2 2026 revenue was a record $28.236 billion, up 26% year-over-year, though EPS of $0.33 missed consensus
- Tesla's Q2 2026 deliveries hit a Q2 record of 480,126 vehicles
- Boeing's Q2 2026 (reported July 28, 2026) revenue was $24.6 billion with a net loss of $428 million
- The FAA restored Boeing's authority to self-issue final airworthiness certificates for 737 MAX/787 jets effective July 20, 2026
- JPMorgan Chase's Q2 2026 EPS was $6.14 (reported July 14, 2026), beating consensus
- Goldman Sachs reported its highest-ever quarterly profit in Q2 2026, with diluted EPS of $20.98
- Delta Air Lines' Q2 2026 adjusted EPS was $1.56 (reported July 10, 2026)
- GM's Q2 2026 revenue was $48.0 billion with net income of $1.3 billion (reported July 21, 2026)
- Iberdrola agreed to take control of Finnish utility Caruna for $2.3 billion (announced July 22, 2026)
- Var Energi agreed to buy BlueNord for $1.3 billion (July 21, 2026), creating Europe's largest independent oil and gas company
- Qualcomm agreed to acquire AI software startup Modular for $3.92 billion (announced June 24, 2026)
- ON Semiconductor announced a $7 billion all-stock acquisition of Synaptics on June 25, 2026
- PwC forecast on June 23, 2026 that global M&A deal value was on track to reach $4 trillion for the year
- The OCC issued a 39-page GENIUS Act stablecoin rulemaking proposal on June 22, 2026
- Regulators missed the GENIUS Act's original July 18, 2026 statutory deadline for finalizing stablecoin rules
- Nvidia implemented a new "whitelist" compliance system for Asian AI-chip customers on July 14, 2026, disqualifying over half of prior eligible buyers
- The DOJ approved the $110 billion Paramount Skydance-Warner Bros. Discovery merger around June 12, 2026, without requiring divestitures
- A coalition of 12 state attorneys general filed a lawsuit July 13, 2026 to block the $110 billion Paramount-WBD merger
- A California federal judge issued a temporary restraining order on July 20, 2026 delaying the Paramount-WBD merger
- Paramount Skydance agreed to halt the Warner Bros. Discovery merger until as late as June 1, 2027, pending an antitrust ruling
- Philippine Airlines ordered 15 Boeing 787-10 Dreamliners at the Farnborough International Airshow (July 2026)
- Uganda Airlines placed its first-ever Boeing order at Farnborough 2026: four 737 MAX 8s and four 787-9 Dreamliners
- Trump announced an additional 50% tariff on Canada on July 20, 2026, effective August 19, 2026
- Trump imposed Section 301 tariffs of 10%-12.5% on more than 60 countries on July 24, 2026
- The EU Council formally adopted regulations enacting the US-EU trade deal on June 25, 2026, effective July 1, 2026 (15% US tariff on most EU goods)
- New UK PM Andy Burnham named John Healey as Chancellor of the Exchequer (July 20, 2026)
- Anthropic filed confidentially for an IPO on June 1, 2026, at a $965 billion valuation after closing a $65 billion Series H round
- The Bank of England held Bank Rate at 3.75% on June 17, 2026, voting 7-2, with two dissenters wanting a hike to 4.00%
- The Swiss National Bank left its policy rate unchanged at 0% on June 18, 2026
- The Bank of Canada held its overnight rate target at 2.25% on July 15, 2026, its sixth consecutive hold
- Alphabet's Q2 2026 revenue was $119.8 billion, up 24% year-over-year, reported July 22, 2026
- Alphabet raised its 2026 capex guidance to as high as $205 billion, up from a prior $180-190 billion forecast
- Bank of America's Q2 2026 net income was $9.1 billion (+27% year-over-year), reported July 14, 2026
- Citigroup's Q2 2026 revenue was $24.8 billion, its best quarterly revenue in a decade, reported July 14, 2026
- Wells Fargo's Q2 2026 investment banking fees set a quarterly record above $900 million, reported July 14, 2026
- Morgan Stanley's Q2 2026 net revenue was $21.35 billion (+27% year-over-year); its Wealth/Investment Management client assets hit a $10 trillion milestone
- Southwest Airlines' Q2 2026 adjusted operating revenue of $8.7 billion was the highest in company history
- UPS's Q2 2026 revenue was $22.8 billion, reported July 28, 2026, though shares fell about 4% despite beating estimates
- FedEx changed its fiscal year end from May 31 to December 31, effective June 1, 2026, and completed the spin-off of FedEx Freight
- Coca-Cola's Q2 2026 net revenue was $13.38 billion (+7% year-over-year), reported July 28, 2026, with shares rising over 7% to a record high
- PepsiCo's Q2 2026 revenue was $24.18 billion (+6.4% year-over-year), reported July 9, 2026
- PayPal's Q2 2026 revenue was $8.68 billion (+5% year-over-year), reported July 28, 2026
- 372 larger US companies filed for bankruptcy protection in the first half of 2026, the highest first-half total since 2010
- President Trump signed a proclamation on June 1, 2026 further adjusting Section 232 tariffs on aluminum, steel, and copper
- Gold fell below $4,000/oz in late June 2026, a steep pullback from January 2026's record above $5,500/oz
- The US dollar hit its best 2026 exchange rate against the yen, 162.6103, on June 30, 2026
- Crypto hacks/losses totaled $75.87 million across 40 incidents in June 2026, with the Humanity Protocol breach the largest single incident at over $30 million
- Brazil's central bank cut the Selic rate to 14.25% at its June 16-17, 2026 meeting, its third consecutive quarter-point cut
- Mexico's Banxico held its rate at 6.50% on June 25, 2026, unanimously, after two years of cuts
- South Korea's central bank raised its base rate to 2.75% on July 16, 2026, its first hike in roughly 3.5 years
- South Africa's central bank held its repo rate at 7% on July 23, 2026, surprising analysts who expected a hike
- Russia's central bank cut its key rate to 14.25% on June 19, 2026, then to 14.00% on July 24, 2026
- South Korea's Kospi index plunged roughly 10% on July 28, 2026, its weakest close since April, on chipmaking-stock selling
- IBM shares crashed 25.2% on July 14, 2026, the worst single day in company history, after missing Q2 earnings estimates
- The "Magnificent Seven" tech stocks lost a combined $797 billion in market value on July 23, 2026, their worst single day since April 2025
- US nonfarm payrolls rose just 57,000 in June 2026 (released July 2, 2026); unemployment held at 4.2%
- US housing starts jumped 19% to a seasonally adjusted 1,427,000 units in June 2026, driven by multifamily construction
- 3M raised its full-year adjusted EPS guidance to $8.80-$8.95 after Q2 2026 results reported July 21, 2026
- ExxonMobil pre-announced on June 7, 2026 that it expected a Q2 profit increase of approximately $3.7 billion from the oil price surge
- The EU-US trade framework formally entered into force July 1, 2026, eliminating tariffs on most US-origin industrial goods
- Gulf sovereign wealth funds invested a record $53.9 billion across 108 transactions in H1 2026, nearly half directed to the US
- The IMF's July 2026 World Economic Outlook Update projected global growth at 3.0% for 2026, with inflation revised up to 4.7%
- 16 startups reached unicorn ($1B+) status in June 2026, per Crunchbase tracking
""",

"PCQ-WOR": """
World Events, June-July 2026:
- Venezuela earthquakes: magnitude 7.2 and 7.5 struck northwestern Venezuela on June 24, 2026
  - 1,430 confirmed dead, thousands more missing, widespread damage
- UK leadership change: Keir Starmer ousted by Labour Party, Andy Burnham became 7th UK PM
  in a decade (July 20, 2026); Burnham's career shaped by England's North-South divide
- US-Iran War: conflict over Strait of Hormuz intensified (July 2026)
  - US and Iran exchanged strikes on infrastructure/military targets July 18, 2026
  - Iran controlled Strait of Hormuz; ~20% of world oil affected
- Taylor Swift married Travis Kelce at Madison Square Garden, July 3, 2026
  - ~1,000 guests attended; thousands of fans gathered outside
- USA celebrated 250th anniversary (semiquincentennial) on July 4, 2026
  - Time capsule buried at Philadelphia's Independence National Historic Park
- Armenia: parliament passed law requiring stricter residency rules for overseas voters
- Egypt advanced to FIFA World Cup Round of 16 for first time (beat Australia on penalties)
- Lionel Messi: first player to make 30 World Cup appearances, scored in 8 consecutive World Cups
- Moldovan PM Alexandru Munteanu resigned
- Sri Lanka prison riots: 27 dead including 4 officers (Negombo prison, rival drug gangs)
- 2026 NATO Summit held Ankara, Turkey July 7-8
- English actress Dame Penelope Keith died June 29, 2026, age 86
- Argentine singer/musician Daniel Melingo died June 30, 2026, age 69, in Buenos Aires
- Fantasy/sci-fi illustrator John Blanche (Warhammer 40,000) died in June 2026, age 77
- Actor Hal Williams, known for the sitcom "227," died July 15, 2026, age 91
- Spoken-word artist Black Ice (Def Poetry Jam) died July 22, 2026, age 54
- Montreal jazz pianist Oliver Jones died July 22, 2026, age 91
- A magnitude 7.8 earthquake struck off Maasim, Sarangani province, Philippines on June 8, 2026, triggering a tsunami alert
- Flash flooding in Afghanistan in June 2026 killed over 300 people
- A burnover incident at the Snyder Fire (Mesa County, Colorado) on June 27, 2026 killed three federal wildland firefighters
- Climate-driven wildfires had displaced more than 300,000 people in France and Spain by July 26, 2026
- The "Thumb Fire" in Minnesota's Superior National Forest started July 7, 2026 and grew to about 14,500 acres
- July 2026 heat domes killed at least 70 people across the US, per a Washington Post analysis
- A PAC P-750 XSTOL skydiving aircraft crashed in Butler, Missouri on June 14, 2026, killing all 12 occupants
- A US Air Force B-52 bomber crashed shortly after takeoff at Edwards Air Force Base, California, on June 15, 2026, killing all 8 crew
- A Cessna 680A Citation Latitude crashed onto Loop 20 highway in Laredo, Texas on June 16, 2026, killing 1 of 6 aboard
- A Pilatus PC-6 Porter carrying skydivers crashed at Nancy-Essey Airport in Tomblaine, France on June 28, 2026, killing all 11 aboard, the deadliest skydiving plane crash in French history
- A shooting on Texas State Highway Loop 250 in Midland, Texas on June 12, 2026 killed 1 person and injured 10
- A shooting at a youth welfare center in Stade, Germany on June 29, 2026 killed 6 people
- A Philadelphia jury found Keith Gibson ("The Beast") responsible for 4 murders on June 9, 2026
- Gilgo Beach serial killer Rex Heuermann was sentenced to life without parole on June 17, 2026, for 8 murders
- A van attack at Berlin's Christopher Street Day (Pride) celebration on July 25, 2026 killed one woman; the perpetrator, Abdul Ballout, was fatally shot by police the next day
- Sakurajima volcano in Japan had one of its most intense eruptions on June 7, 2026
- Kilauea's Halema'uma'u eruption Episode 49 occurred June 14, 2026, with lava fountains reaching almost 700 feet
- A new Mount Etna eruption began June 26, 2026, with lava flowing toward the Valle del Leone
- The 2026 Tony Awards were held June 7, 2026 at Radio City Music Hall
- "Schmigadoon!" won Best Musical at the 2026 Tony Awards
- "Liberation" won Best Play at the 2026 Tony Awards
- "Death of a Salesman" won Best Revival of a Play at the 2026 Tony Awards (6 Tonys total)
- "Ragtime" won Best Revival of a Musical at the 2026 Tony Awards
- John Lithgow won Best Actor in a Play at the 2026 Tony Awards, for "Giant"
- Lesley Manville won Best Actress in a Play at the 2026 Tony Awards, for "Oedipus"
- 2026 Emmy nominations were announced July 8, 2026; HBO Max's "The Pitt" led with 25 nominations
- "Hacks" received 24 Emmy nominations in 2026, including Outstanding Lead Actress for Jean Smart
- The 2026 Booker Prize longlist was announced July 28, 2026
- A cache of 43 helmets found off the Spanish coast was confirmed in June 2026 to be medieval, not Roman as previously thought
- Mexico's INAH announced a newly discovered 1,400-year-old archaeological site in Coatepec, Veracruz, around June 29, 2026
- Archaeologists revealed a Greco-Roman cemetery atop older settlement remains at Tel Kom Aziza, Egypt, in June 2026
- A prehistoric painted cave with nearly 100 human/animal figures was identified in Malatya's Tohma Canyon, Turkey, in June 2026
- A 5-year-old girl, Daleyza Fregoso, subject of an AMBER Alert, was found safe in Mexico on June 15, 2026
- A 93-year-old Hoke County, NC woman, Catherine Peterkin, reported missing, was found safe on June 19, 2026
- Welsh singer Bonnie Tyler ("Total Eclipse of the Heart") died unexpectedly July 9, 2026, in a hospital in Portugal
- Village People co-founder and lead singer Victor Willis died June 30, 2026, age 74
- Grammy-winning R&B singer Peabo Bryson, known for Disney's "Beauty and the Beast"/"Aladdin" theme songs, died June 2, 2026
- Actor Anthony Head, known for "Buffy the Vampire Slayer" and "Ted Lasso," died June 5, 2026 at age 72
- Brenda Fricker, the first Irish actress to win an Oscar ("My Left Foot," 1989), died July 16, 2026 in Dublin at age 81
- Belgian Nobel physics laureate Francois Englert died June 18, 2026 in Uccle, Belgium, at age 93
- Twin strike-slip earthquakes struck northwestern and central Venezuela on June 24, 2026, killing at least 164 people
- A landslide at Renzang village, Longnan City, Gansu Province, China on July 7, 2026 killed 21 people
- A landslide on the Wujiang River at Hanjia, Chongqing, China on July 17, 2026 killed at least 8 people with 34 missing
- The ferry MV Barima capsized and sank on July 18, 2026 en route from Georgetown to Port Kaituma, Guyana, with 73 confirmed dead, called Guyana's worst maritime disaster on record
- A coach bus flipped over the median into oncoming traffic on the Long Island Expressway in Queens, NY around 11:45pm on June 29, 2026, killing the driver and a passenger
- An explosion at Hanwha Aerospace's Daejeon, South Korea plant on June 1, 2026 killed 5 workers
- A crush at Hashemite Square in Amman, Jordan on June 23, 2026 during a public World Cup screening killed 1 person and injured 8
- Teyana Taylor won four awards including Icon of the Year at the BET Awards 2026 (held June 28, 2026, Peacock Theater, Los Angeles)
- Cardi B won Best Female Hip-Hop Artist and Kendrick Lamar won Best Male Hip-Hop Artist at the 2026 BET Awards
- The Guggenheim Museum in New York opened "Guggenheim Pop," a major survey of 20th-century Pop art, in June 2026
- Red panda twins were born June 3, 2026 to parents Nila and Ash at Hertfordshire Zoo, UK, its first red panda twins in 12 years
- Idaho's David Rush balanced 195 rolls of toilet paper on his head for 30 seconds to recapture a Guinness World Record, reported July 20, 2026
- A fossilized Edmontosaurus skull with a Tyrannosaurus tooth still embedded in its face was announced July 14, 2026, rare direct evidence of a predator-prey attack
- A Dutch archaeological mission from Leiden University discovered a previously unknown 3,000-year-old tomb belonging to a man named Paser near Luxor, Egypt, announced July 14, 2026
- 1992 Nobel Chemistry laureate Rudolph A. Marcus died July 16, 2026 at Caltech, aged 102
- Saxophonist Plas Johnson, who played the famous sax line on "The Pink Panther" theme, died July 15, 2026 in Los Angeles, aged 94
- Olympic cycling silver medalist Toshiaki Fushimi died July 19, 2026 of injuries from a keirin race crash in Matsusaka, Japan, aged 50
- The 2026 Kumamoto earthquake struck Kyushu Island, Japan on July 28, 2026 (magnitude 6.8)
- Super Typhoon Bavi passed over the Northern Mariana Islands around July 5-6, 2026 before making landfall in Zhejiang, China on July 11
- Typhoon Noul made landfall in Huidong county, Guangdong province, China at about 3:50am local time on July 26, 2026
- Two passenger trains collided near Bedford, England on June 19, 2026, killing one driver and hospitalizing more than 80 people
- A giant panda cub was born June 3, 2026 at Panda World, Everland Zoo, South Korea, to parents Ai Bao and Le Bao
- The Pentagon released its third batch of declassified UAP files on June 12, 2026 (53 documents, 10 images, 6 videos)
- Pope Leo XIV made an apostolic journey to Spain June 6-12, 2026, inaugurating the Tower of Jesus Christ at the Sagrada Familia
- King Charles III led the Order of the Garter ceremony at St George's Chapel, Windsor Castle, on June 15, 2026
- France's Fete de la Musique (June 21, 2026) saw over 240 arrests nationwide per the interior ministry
- The US Department of Energy issued an emergency order activating dormant power sources across 17 states due to heat straining the grid, in effect through August 3, 2026
"""
}

PCQ_DOMAINS = {
    "PCQ-SPO": "sports",
    "PCQ-POL": "politics and government",
    "PCQ-SCI": "science and technology",
    "PCQ-ECO": "economy and business",
    "PCQ-WOR": "world events",
}

SEED = 42


# ----------------------------------------------------------------------
# Veri yapisi (mevcut EHQ semasiyla uyumlu)
# ----------------------------------------------------------------------

@dataclass
class PCQItem:
    question_id:    str
    category:       str  = "PCQ"
    subcategory:    str  = ""
    question:       str  = ""
    correct_answer: str  = ""
    event_date:     str  = ""
    source_fact:    str  = ""   # gold answer'in dayandigi kaynak bilgisi
    qc_passed:      bool = False
    qc_notes:       list = field(default_factory=list)


# ----------------------------------------------------------------------
# Uretici: Mistral Large (ASU)
# ----------------------------------------------------------------------

_GEN_PROMPT = """You are generating factual quiz questions for an AI benchmark.

Based on the following verified facts about events in {window}, generate ONE question that:
1. Has a single, specific, unambiguous factual answer
2. Can be answered from the facts below WITHOUT needing external information
3. Is NOT answerable from general knowledge about events BEFORE {window}
   (it must be about a specific result, name, number, date, or decision from this period)
4. Has a concise gold answer (1-10 words maximum)

VERIFIED FACTS:
{facts}

Generate a different question each time - vary the topic, entity, and type of fact asked.

Return STRICT JSON ONLY (no markdown):
{{
  "question": "<specific factual question about {window} events>",
  "correct_answer": "<concise factual answer, 1-10 words>",
  "event_date": "<approximate date within {window}>",
  "source_fact": "<the specific fact from above that supports this answer>"
}}"""


def _gen_call(prompt: str, temperature: float = 0.8) -> Optional[str]:
    # use_cache=False: ayni fact bloguyla tekrar tekrar cagriliyoruz ve HER
    # seferinde FARKLI bir soru istiyoruz (temperature=0.8 rastgeleligi
    # bunun icin var). asu_client'in diski cache'i (model+prompt+query hash)
    # temperature'i anahtara katmiyor -- prompt ayni kaldiginda (bir alt
    # kategoride art arda "duplicate question" ile reddedilen denemeler
    # seen_questions'i buyutmedigi surece prompt degismez) her cagriyi
    # ayni cache'lenmis yanita yonlendirip sonsuz ayni-soru dongusune
    # sokuyordu (900 denemede tek soru). Cache burada kapatilmali.
    return asu_query(
        model_name=GENERATOR_MODEL,
        model_provider=GENERATOR_PROVIDER,
        query=prompt,
        temperature=temperature,
        request_delay=1.5,
        use_cache=False,
    )


def generate_raw_item(subcode: str, existing_questions: set) -> Optional[dict]:
    """Kaynak havuzundan tek PCQ uret."""
    facts = SOURCE_FACTS[subcode]
    # Tekrar onlemek icin mevcut sorulari prompt'a ekle
    avoid = ""
    if existing_questions:
        sample = list(existing_questions)[-10:]  # (seen_questions kategoriler arasi paylasilir)
        avoid = f"\n\nAVOID generating questions similar to these already generated:\n" + \
                "\n".join(f"- {q}" for q in sample)
    
    prompt = _GEN_PROMPT.format(
        window=EVENT_WINDOW,
        facts=facts + avoid,
    )
    raw = _gen_call(prompt)
    if not raw:
        return None
    cleaned = re.sub(r"```(json)?", "", raw).strip()
    try:
        data = json.loads(cleaned)
        if all(k in data for k in ("question", "correct_answer",
                                    "event_date", "source_fact")):
            return data
    except json.JSONDecodeError:
        logger.warning("JSON parse failed for %s", subcode)
    return None


# ----------------------------------------------------------------------
# QC filtreleri
# ----------------------------------------------------------------------

MAX_ANSWER_REUSE = 2   # ayni gercek/cevaptan en fazla 2 soru uretilebilir


def run_qc(item: PCQItem, seen_questions: set, answer_counts: dict) -> PCQItem:
    notes = []
    if len(item.question.split()) < 5:
        notes.append("question too short")
    if not item.correct_answer or len(item.correct_answer.strip()) < 1:
        notes.append("empty correct_answer")
    if len(item.correct_answer.split()) > 15:
        notes.append("answer too long (>15 words)")
    # Tekrar kontrolu (seen_questions/answer_counts TUM kategoriler arasinda
    # paylasilir; farkli alt kategoriler ayni kaynak olayi sormasin)
    q_norm = item.question.lower().strip()
    if q_norm in seen_questions:
        notes.append("duplicate question")
    ans_norm = item.correct_answer.lower().strip()
    if answer_counts.get(ans_norm, 0) >= MAX_ANSWER_REUSE:
        notes.append(f"answer reused >{MAX_ANSWER_REUSE}x (same underlying fact)")
    # Cevap soruda gizleniyor mu?
    if item.correct_answer.lower() in item.question.lower():
        notes.append("answer appears in question")
    item.qc_notes = notes
    item.qc_passed = len(notes) == 0
    return item


# ----------------------------------------------------------------------
# Tek uretim + toplu uretim
# ----------------------------------------------------------------------

def build_one(subcode: str, idx: int,
              seen_questions: set, answer_counts: dict) -> Optional[PCQItem]:
    raw = generate_raw_item(subcode, seen_questions)
    if not raw:
        return None
    item = PCQItem(
        question_id=f"{subcode}-{idx:03d}",
        subcategory=subcode,
        question=raw["question"].strip(),
        correct_answer=raw["correct_answer"].strip(),
        event_date=raw["event_date"].strip(),
        source_fact=raw["source_fact"].strip(),
    )
    item = run_qc(item, seen_questions, answer_counts)
    status = "PASS" if item.qc_passed else f"FAIL({'; '.join(item.qc_notes)})"
    logger.info("[%s] %s | Q: %s", item.question_id, status,
                item.question[:60])
    return item


def build_dataset(per_subcategory: int = 150,
                  out_path: str = "PCQ_dataset.json") -> list:
    """EHQ-3000 nihai hedefi 5 x 150 = 750'dir. SOURCE_FACTS havuzu (en dar
    kategori PCQ-WOR icin ~89 benzersiz gercek) uc arastirma turuyla
    77->472 gercege genisletildi; her gercekten en fazla MAX_ANSWER_REUSE
    (2) farkli soru turetilmesine izin verilerek (run_qc/answer_counts)
    per_subcategory=150 hedefi artik ulasilabilir sinirin icinde."""
    all_items = []
    # Kategoriler arasi paylasilan yapilar: SOURCE_FACTS kategorileri ortak
    # olaylar icerdiginden (orn. Venezuela depremi PCQ-ECO ve PCQ-WOR'da da
    # var), dedup tek bir alt kategoriyle sinirli kalamaz.
    seen_q = set()
    answer_counts: dict = {}
    for subcode in PCQ_DOMAINS:
        collected, attempts = 0, 0
        max_attempts = per_subcategory * 6
        while collected < per_subcategory and attempts < max_attempts:
            attempts += 1
            item = build_one(subcode, collected + 1, seen_q, answer_counts)
            if item and item.qc_passed:
                seen_q.add(item.question.lower().strip())
                ans_norm = item.correct_answer.lower().strip()
                answer_counts[ans_norm] = answer_counts.get(ans_norm, 0) + 1
                all_items.append(item)
                collected += 1
        logger.info("Subcategory %s: %d/%d in %d attempts",
                    subcode, collected, per_subcategory, attempts)

    serialised = [asdict(it) for it in all_items]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serialised, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d PCQ items -> %s", len(serialised), out_path)
    return all_items


if __name__ == "__main__":
    import argparse
    import random
    random.seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Tam uretim: 5 alt kategori x 150 = 750 hedef "
                             "(EHQ-3000 nihai hedefi; her gercekten en fazla "
                             "2 soru turetilerek PCQ_dataset.json'a yazilir). "
                             "Verilmezse smoke test calisir.")
    args = parser.parse_args()

    logger.info("Pencere: %s | Uretici: Mistral-Large(ASU)", EVENT_WINDOW)
    logger.info("Kaynak: Gercek olaylar (FIFA WC, NATO, AI/Tech, IMF, Dunya)")
    if not os.environ.get("ASU_CREATEAI_TOKEN"):
        logger.info("ASU_CREATEAI_TOKEN yok; import OK.")
    elif args.full:
        logger.info("PCQ TAM URETIM | 5 alt kategori x 150 = 750 hedef")
        build_dataset(per_subcategory=150, out_path="PCQ_dataset.json")
    else:
        logger.info("PCQ smoke test | Her kategoriden 2 soru")
        build_dataset(per_subcategory=2, out_path="PCQ_smoke.json")
