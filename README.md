# phocus

![](https://lightroom-photoshop-tutorials.com/wp-content/uploads/2020/08/The-Art-of-Focus-in-Photography-Capturing-Clarity-Amidst-Chaos.webp)

getting the LLM to pay attention at all costs.

---

# Overcoming the "Sins of Attention" 

At the heart of LLMs lies the **attention mechanism**, a computational strategy that allows models to weigh the importance of different words in a sentence relative to each other. This mechanism enables models to capture contextual relationships and generate coherent, contextually relevant responses.

LLMs, are **pretrained on vast and diverse datasets** encompassing a wide array of topics and formats. However, when these models are deployed for specific tasks—such as extracting movie titles from a database—they often encounter scenarios that diverge from their training data. This discrepancy necessitates the model to **make precise abstractions** from potentially **noisy or irrelevant inputs**. 

**Attention mechanisms** are at the heart of this capability, enabling models to **focus on relevant information** while disregarding the rest. However, despite their pivotal role, **attention mechanisms are rarely the primary target for optimization** in model training. Instead, the emphasis is typically on overall performance metrics like accuracy and coherence. This oversight can lead to scenarios where the attention mechanism fails to prioritize critical information effectively, resulting in the "sins of attention" discussed earlier.

### The Concept of Attention

The concept of attention in LLMs is inspired by human cognition—specifically, how we selectively concentrate on certain aspects of information while ignoring others. This is similar to how when we read or listen, we don't pay equal attention to every word; instead, we focus more on the words that are most relevant to the overall meaning.

### Vocabulary and Attention

![](https://miro.medium.com/v2/resize:fit:1050/1*UUkr5OyPj-dDl8OaPb-Vow.png)

LLMs are trained on a vast amount of text data, during which they build a large vocabulary of words and phrases. Each word in the model's vocabulary is represented as a high-dimensional vector, which captures its semantic meaning based on the contexts in which it appears in the training data.

### How Attention Works in LLMs

![](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F2d632b81-5bd6-432a-a456-37f20788be20_1180x614.png)

In LLMs, the attention mechanism works by assigning different weights to different parts of the input. These weights determine how much 'attention' the model pays to each part of the input when generating each part of the output.

Consider a more complex sentence translation task. When translating the English sentence "The quick brown fox jumps over the lazy dog" to Spanish ("El rápido zorro marrón salta sobre el perro perezoso"), the model needs to 'pay attention' to different words in the English sentence when generating each word in the Spanish sentence. 

For instance, when generating the word "rápido", the model needs to pay more attention to the word "quick" in the English sentence. Similarly, when generating "zorro", it needs to focus more on "fox", and so on. 

In this context, 'paying attention' means that the model assigns a higher weight to the relevant word in the input when generating each word in the output. These weights are determined based on the learned relationships between words in the model's vocabulary.

This ability to focus on relevant parts of the input while generating the output is what makes LLMs so effective at tasks like translation, summarization, and question answering. However, it's not infallible, and can sometimes lead to errors or omissions—the so-called "sins of attention".

## [The "Sins of Attention"](https://github.com/ranfysvalle02/lost-in-the-middle)

While attention mechanisms are pivotal for LLM performance, they are not infallible. The term **"sins of attention"** refers to various ways in which attention mechanisms can falter, leading to:

1. **Omission of Critical Information:** The model might overlook essential details, leading to incomplete or incorrect responses.
2. **Contextual Misalignment:** The model may misinterpret the context, associating words or phrases incorrectly.
3. **Format Deviations:** Responses may deviate from the expected format, making them difficult to parse or utilize effectively.
4. **Inconsistency Across Batches:** When processing data in batches, the model might inconsistently handle different segments, leading to variable output quality.

These issues are particularly evident in tasks that require precise data extraction or formatting, such as compiling lists from databases or generating structured reports.

## Case Study: Extracting Movie Titles from a Database using an LLM

### The Challenge

When attempting to extract structured data from a database using an LLM, several issues can arise:

- **Incomplete Data Extraction:** The model might miss some entries, leading to fewer results than expected.
- **Formatting Errors:** The output might not adhere to the required format, making it difficult to parse programmatically.
- **Inconsistent Results:** Different batches of data might yield varying levels of accuracy and completeness.

These challenges can hinder the reliability of applications relying on accurate data extraction.

### The Solution: Verification and Retry Mechanisms

To overcome these challenges, implementing **verification and retry mechanisms** is essential. These mechanisms ensure that the LLM's outputs meet the expected criteria, and in cases where they don't, the system can attempt to rectify the issue automatically.


#### 1. Batch Processing with Verification

Processing data in manageable batches allows for better control and error handling. After each batch is processed, the output is verified to ensure it meets the expected criteria.

```python
def send_request_to_model(context, batch_size, desiredModel, max_retries=10, backoff_factor=0):
    system_message, prompt, user_instructions = prepare_prompt(context, batch_size)
    
    for attempt in range(1, max_retries + 1):
        try:
            response = ollama.chat(
                model=desiredModel,
                messages=[
                    {'role': 'system', 'content': system_message},
                    {'role': 'user', 'content': prompt},
                    {'role': 'user', 'content': user_instructions},
                ]
            )
            
            if response.get('message') and response['message'].get('content'):
                csv_output = response['message']['content']
                csv_lines = [line.strip().strip('"') for line in csv_output.split('\n') 
                             if line.strip() and line.strip() not in {'```', '.', '`'}]
                
                if len(csv_lines) == batch_size:
                    return csv_lines
                else:
                    print(f"Attempt {attempt}: Expected {batch_size} titles, got {len(csv_lines)}.")
            else:
                print(f"Attempt {attempt}: No response content received.")
                
        except Exception as e:
            print(f"Attempt {attempt}: Error communicating with the model: {e}")
        
        if attempt < max_retries:
            sleep_time = backoff_factor ** attempt
            print(f"Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
        else:
            print(f"Attempt {attempt}: Max retries reached.")
    
    return []
```

**Key Points:**

- **Preparation of Prompts:** Crafting precise prompts guides the LLM to produce the desired output format.
- **Verification:** After receiving the response, the script checks if the number of extracted titles matches the expected `batch_size`.
- **Retry Mechanism:** If verification fails, the system retries the request up to `max_retries` times, implementing an exponential backoff strategy to handle transient issues.

#### 2. Handling Failed Batches

Some batches might fail even after all retry attempts. It's crucial to track these failed batches for manual inspection or further automated retries.

```python
failed_batches = []

for i, doc in enumerate(formatted_text, 1):
    tmp_batch.append(doc)
    
    if i % batch_size == 0:
        context = parse_json_to_text(tmp_batch)
        csv_lines = send_request_to_model(context, batch_size, desiredModel)
        
        if csv_lines:
            final_result.extend(csv_lines)
            total_docs += len(csv_lines)
            batch_round += 1
            print(f"Batch {batch_round}: Total documents processed: {total_docs}")
        else:
            failed_batches.append(list(tmp_batch))
            print(f"Batch {batch_round + 1}: Failed to process. Will retry later.")
        
        tmp_batch = []
```

**Key Points:**

- **Tracking Failures:** Failed batches are stored in a `failed_batches` list for later retries.
- **Logging:** Informative print statements provide real-time feedback on processing status.

#### 3. Retrying Failed Batches

After processing all initial batches, the system attempts to reprocess any failed batches.

```python
if failed_batches:
    print(f"\nRetrying {len(failed_batches)} failed batch(es)...")
    remaining_failed_batches = []
    for idx, batch in enumerate(failed_batches, 1):
        print(f"\nRetrying Batch {idx} of {len(failed_batches)}:")
        context = parse_json_to_text(batch)
        current_batch_size = len(batch)
        csv_lines = send_request_to_model(context, current_batch_size, desiredModel)
        
        if csv_lines:
            final_result.extend(csv_lines)
            total_docs += len(csv_lines)
            batch_round += 1
            print(f"Retry Batch {idx}: Total documents processed: {total_docs}")
        else:
            remaining_failed_batches.append(batch)
            print(f"Retry Batch {idx}: Failed again.")
    
    if remaining_failed_batches:
        print(f"\n{len(remaining_failed_batches)} batch(es) failed after retries. Please inspect manually.")
        for idx, batch in enumerate(remaining_failed_batches, 1):
            print(f"\nFailed Batch {idx}:")
            for doc in batch:
                print(doc.get('title', 'N/A'))
    else:
        print("\nAll batches processed successfully after retries.")
```

**Key Points:**

- **Final Attempt:** The system makes a final attempt to process any batches that failed during the initial processing.
- **Manual Inspection:** Batches that fail even after retries are logged for manual intervention, ensuring no data is permanently lost.

## Best Practices

By implementing verification and retry mechanisms, you can significantly enhance the reliability of LLMs in data processing tasks. 

### 1. **Implement Robust Verification**

Always verify the LLM's output against expected criteria, such as the number of items or the format of the data. Automated checks ensure consistency and completeness, reducing the risk of human error.

```python
if len(csv_lines) == batch_size:
    # Success
else:
    # Trigger retry
```

### 2. **Design Effective Retry Strategies**

Implement retry mechanisms with configurable parameters like `max_retries` and `backoff_factor`. Exponential backoff helps manage the timing of retries, reducing the likelihood of overwhelming the model or API.

```python
for attempt in range(1, max_retries + 1):
    # Attempt processing
    if success:
        break
    else:
        sleep_time = backoff_factor ** attempt
        time.sleep(sleep_time)
```

### 3. **Log and Monitor Failures**

Maintain detailed logs of all processing attempts, especially failures. Monitoring allows you to identify patterns or persistent issues that may require manual intervention or further optimization.

```python
print(f"Attempt {attempt}: Error - {e}")
```

### 4. **Handle Partial and Edge Cases Gracefully**

Ensure that your system can handle edge cases, such as the final batch containing fewer items than the standard batch size. Adjust expectations dynamically based on the context.

```python
if tmp_batch:
    # Handle last batch
```

### 5. **Provide Clear Feedback and Logging**

Use informative print statements or logging frameworks to provide real-time feedback on processing status. Clear logging aids in debugging and tracking the system's performance over time.

```python
print(f"Batch {batch_round}: Total documents processed: {total_docs}")
```

## Conclusion

Large Language Models have undeniably transformed the landscape of artificial intelligence, offering unparalleled capabilities in understanding and generating human-like text. However, their reliance on attention mechanisms can sometimes lead to challenges such as data omission, contextual misalignment, and formatting errors—the so-called "sins of attention."

By implementing **robust verification and retry mechanisms**, you can significantly mitigate these issues. 

**Key Takeaways:**

- **Verification is Crucial:** Always verify the LLM's output to ensure it meets expected criteria.
- **Retry Mechanisms Enhance Reliability:** Implementing retries with strategies like exponential backoff can handle transient failures gracefully.
- **Logging and Monitoring Aid in Maintenance:** Detailed logs help in identifying and addressing persistent issues.

As AI continues to advance, developing systems that complement and reinforce LLM capabilities will be crucial in harnessing their full potential while minimizing their shortcomings. Embracing these best practices not only enhances the reliability of AI-driven applications but also empowers you to build more sophisticated and dependable AI systems in the future.

---

### Appendix: Advanced Attention Techniques for Enhanced Data Processing

As we delve deeper into optimizing interactions with Large Language Models (LLMs), understanding and leveraging advanced attention mechanisms can significantly enhance performance, especially when dealing with extensive datasets. This appendix explores cutting-edge techniques like **Longformer**, **Sparse Attention**, and other innovative methods designed to overcome the limitations discussed earlier.

#### **1. [Longformer](https://arxiv.org/abs/2004.05150): Extending the Context Window**

**Longformer** is an extension of the Transformer architecture tailored to handle longer sequences of text efficiently. Traditional Transformers, including models like GPT-4, struggle with very long inputs due to their quadratic complexity in the attention mechanism. Longformer addresses this challenge through the implementation of **sliding window attention**, allowing the model to process longer texts without a proportional increase in computational resources.

**Key Features:**
- **Sliding Window Attention**: Instead of attending to every token in the input, Longformer restricts attention to a fixed-size window around each token. This reduces the computational load and enables the model to handle longer sequences.
- **Global Attention**: Certain tokens can be designated to have global attention, meaning they can attend to all other tokens in the sequence. This is useful for tasks requiring an understanding of the entire context, such as question answering or summarization.

**Benefits:**
- **Scalability**: Efficiently processes longer texts without exhausting memory resources.
- **Flexibility**: Combines local and global attention mechanisms to maintain context where it's most needed.

#### **2. [Sparse Attention: Optimizing Focused Processing](https://arxiv.org/abs/2406.15486)**

**Sparse Attention** mechanisms aim to reduce the computational overhead of processing long sequences by limiting the number of attention connections. Unlike dense attention, where every token attends to every other token, sparse attention introduces patterns that determine which tokens interact, significantly cutting down the number of computations required.

**Key Patterns:**
- **Fixed Patterns**: Predefined attention patterns, such as attending to every nth token or forming a fixed grid.
- **Learned Patterns**: Attention patterns that the model learns during training, allowing for more dynamic and contextually relevant connections.

**Benefits:**
- **Efficiency**: Decreases memory usage and increases processing speed, making it feasible to handle larger inputs.
- **Customization**: Can be tailored to specific tasks, ensuring that the most relevant parts of the input are prioritized.

#### **3. [Reformer: Memory-Efficient Transformers](https://arxiv.org/pdf/2001.04451)**

**Reformer** introduces several innovations to make Transformer models more memory-efficient and faster, enabling them to handle longer sequences without compromising performance.

**Key Innovations:**
- **Locality-Sensitive Hashing (LSH) Attention**: Groups similar tokens together, allowing the model to compute attention within these groups rather than across the entire sequence.
- **Reversible Layers**: Reduces memory usage by allowing intermediate activations to be recomputed during the backward pass, eliminating the need to store them.

**Benefits:**
- **Memory Efficiency**: Significantly reduces the memory footprint, allowing for training and inference on longer sequences.
- **Speed**: Enhances processing speed by optimizing attention computations.

#### **4. [Performer: Linear Attention Mechanisms](https://arxiv.org/abs/2009.14794)**

**Performer** introduces **linear attention**, which scales linearly with the sequence length, as opposed to the quadratic scaling seen in traditional attention mechanisms. This innovation makes it feasible to handle very long sequences with reduced computational complexity.

**Key Features:**
- **FAVOR+ (Fast Attention Via positive Orthogonal Random features)**: An approximation technique that allows attention to be computed more efficiently without significant loss of accuracy.
- **Kernel-based Attention**: Transforms the attention computation into a kernel function, facilitating faster processing.

**Benefits:**
- **Scalability**: Easily handles long sequences, making it suitable for tasks like document processing and large-scale data analysis.
- **Performance**: Maintains high accuracy while significantly reducing computational requirements.

#### **5. [Memory-Augmented Networks: Extending Model Capacity](https://arxiv.org/html/2312.06141v2)**

**Memory-Augmented Networks** integrate external memory components with LLMs, allowing them to store and retrieve information beyond their inherent context window. This approach effectively extends the model's capacity to handle larger datasets without overloading the attention mechanism.

**Key Components:**
- **External Memory Banks**: Structured storage that the model can read from and write to, enabling persistent storage of information.
- **Read/Write Operations**: Mechanisms that allow the model to access relevant information from the external memory as needed.

**Benefits:**
- **Extended Context**: Enables models to reference a much larger set of data without processing it all simultaneously.
- **Improved Accuracy**: Enhances the model's ability to recall and utilize information effectively, leading to more accurate and comprehensive outputs.

#### **6. Retrieval-Augmented Generation (RAG): Enhancing Contextual Understanding**

**Retrieval-Augmented Generation (RAG)** combines traditional language models with retrieval systems to fetch relevant information from external databases or documents in real-time. This hybrid approach allows models to access a vast pool of knowledge without being constrained by their fixed context window.

**Key Features:**
- **Dual Components**: Combines a retrieval system (e.g., a search engine) with a generative model.
- **Dynamic Information Access**: Fetches relevant data on-the-fly based on the input query or context.

**Benefits:**
- **Up-to-Date Information**: Allows models to access the latest information beyond their training data.
- **Enhanced Accuracy**: Improves response relevance by grounding generation in retrieved data.

#### **7. Hybrid Models: Combining Strengths for Optimal Performance**

**Hybrid Models** integrate multiple attention mechanisms or combine Transformers with other neural network architectures to leverage the strengths of each. By doing so, they aim to balance computational efficiency with comprehensive data processing capabilities.

**Key Strategies:**
- **Combining Sparse and Dense Attention**: Utilizes sparse attention for most of the input while applying dense attention to critical sections.
- **Integrating Convolutional Layers**: Adds convolutional layers to capture local patterns before passing data to the Transformer layers.

**Benefits:**
- **Balanced Performance**: Achieves a middle ground between efficiency and thoroughness.
- **Task-Specific Optimization**: Tailors the model architecture to better suit specific application needs.

### **Leveraging Advanced Attention Techniques**

The challenges posed by attention mechanisms in LLMs, such as data being "lost in the middle," are significant but not insurmountable. By embracing advanced techniques like Longformer, Sparse Attention, Reformer, Performer, Memory-Augmented Networks, Retrieval-Augmented Generation, and Hybrid Models, developers can enhance the capability of their models to handle large and complex datasets more effectively.

These innovations not only address the limitations of traditional attention mechanisms but also open new avenues for creating more robust, efficient, and versatile AI systems. As the field of artificial intelligence continues to advance, staying informed about these cutting-edge techniques will empower you to optimize your workflows, ensuring that your models can process and retain the vast amounts of data they encounter without losing valuable information along the way.

---

## OUTPUT

```
Batch 50: Total documents processed: 1000

All 1000 movie titles have been successfully collected.

Here is the complete list of movie titles:
1: The Perils of Pauline
2: Traffic in Souls
3: The Great Train Robbery
4: Civilization
5: The Poor Little Rich Girl
6: Wild and Woolly
7: The Four Horsemen of the Apocalypse
8: Now or Never
9: Beau Geste
10: Storm Over Asia
11: Steamboat Willie
12: Applause
13: In Old Arizona
14: A Free Soul
15: Love Me Tonight
16: Cavalcade
17: Duck Soup
18: Eskimo
19: Queen Christina
20: Chapayev
21: Imitation of Life
22: Triumph of the Will
23: Twentieth Century
24: The World Moves On
25: David Copperfield
26: The Devil Is a Woman
27: The Invisible Ray
28: Mary of Scotland
29: Mayerling
30: Rembrandt
31: Marie Antoinette
32: You Can't Take It With You
33: Of Mice and Men
34: The Story of Vernon and Irene Castle
35: Wuthering Heights
36: Kitty Foyle
37: Mrs. Miniver
38: This Above All
39: Air Force
40: Cabin in the Sky
41: Leopard Man
42: Hail the Conquering Hero
43: Torment
44: Laura
45: The Way Ahead
46: L'espoir
47: Children of Paradise
48: Isle of the Dead
49: Leave Her to Heaven
50: The Bandit
51: Devil in the Flesh
52: Levoton veri
53: Song of the South
54: To Each His Own
55: A Double Life
56: It Happened on Fifth Avenue
57: Where Are My Children?
58: The Saphead
59: Miss Lulu Bett
60: Robin Hood
61: Westfront 1918
62: The Big House
63: Dracula
64: The Guardsman
65: Comradeship
66: The Public Enemy
67: è Nous la Libertè
68: The Private Life of Henry VIII.
69: Three Little Pigs
70: Topaze
71: Maria Chapdelaine
72: Les Misèrables
73: Moscow Laughs
74: Broadway Melody of 1936
75: Dante's Inferno
76: Les Misèrables
77: She
78: The Prisoner of Shark Island
79: These Three
80: Tsirk
81: Elephant Boy
82: Harvest
83: The Hurricane
84: Kid Galahad
85: Children in the Wind
86: Make Way for Tomorrow
87: Shall We Dance
88: The Cowboy and the Lady
89: Olympia Part Two: Festival of Beauty
90: The Beachcomber
91: Destry Rides Again
92: Gone with the Wind
93: Zangiku monogatari
94: Arise, My Love
95: Jud Sèè
96: The Long Voyage Home
97: The Land
98: ''Pimpernel'' Smith
99: They Met in Moscow
100: Casablanca
101: Meshes of the Afternoon
102: Domingo de carnaval
103: Gertie the Dinosaur
104: Muzi bez krèdel
105: The Blue Bird
106: The Fighting Lady
107: The Life and Death of Colonel Blimp
108: This Happy Breed
109: Tol'able David
110: Hungry Hill
111: Detour
112: Sister Kenny
113: State Fair
114: The Jolson Story
115: Doubling for the Enemy
116: Regeneration
117: Lassie Come Home
118: The Major and the Minor
119: Caesar and Cleopatra
120: Ziegfeld Follies
121: A Woman of Paris: A Drama of Fate
122: The Iron Horse
123: White Shadows
124: The Broadway Melody
125: Hallelujah
126: Queen Kelly
127: The Devil to Pay!
128: The Front Page
129: Tabu: A Story of the South Seas
130: The Red Head
131: The Man Who Knew Too Much
132: One Night of Love
133: Everybody's Woman
134: Toni
135: The Dark Angel
136: The Ghost Goes West
137: Gold Diggers of 1935
138: Wife! Be Like a Rose!
139: The Beloved Vagabond
140: Der Kaiser von Kalifornien
141: Final Accord
142: Lenin in October
143: The Pearls of the Crown
144: The Prince and the Pauper
145: Waikiki Wedding
146: Love Finds Andy Hardy
147: Olympia Part One: Festival of the Nations
148: The Rage of Paris
149: White Banners
150: Confessions of a Nazi Spy
151: La fin du jour
152: The Four Feathers
153: Dots
154: The Mark of Zorro
155: North West Mounted Police
156: The Westerner
157: Dumbo
158: Major Barbara
159: The White Ship
160: The Sea Wolf
161: Across the Pacific
162: Black Swan
163: Holiday Inn
164: Jungle Book
165: Journey for Margaret
166: Woman of the Year
167: Bataan
168: The Ghost Ship
169: Shadow of a Doubt
170: The Song of Bernadette
171: Gaslight
172: The House on 92nd Street
173: Red Meadows
174: Story of G.I. Joe
175: A Tree Grows in Brooklyn
176: The Valley of Decision
177: The Battle of the Rails
178: Murderers Among Us
179: The Razor's Edge
180: Rome, Open City
181: Winsor McCay, the Famous Cartoonist of the N.Y. Herald and His Moving Comics
182: In the Land of the Head Hunters
183: Ella Cinders
184: It
185: Wings
186: Men Without Women
187: Morocco
188: Romance
189: L'opèra de quat'sous
190: The Blue Light
191: The Music Box
192: Payment Deferred
193: Red Dust
194: The Son of Kong
195: The Barretts of Wimpole Street
196: The Black Cat
197: It's a Gift
198: Viva Villa!
199: Wonder Bar
200: Anna Karenina
201: The Gilded Lily
202: The Informer
203: A Midsummer Night's Dream
204: Black Legion
205: The Charge of the Light Brigade
206: Dodsworth
207: The Garden of Allah
208: The Green Pastures
209: Three Smart Girls
210: Daughter of Shanghai
211: The Edge of the World
212: Angels with Dirty Faces
213: The Big Broadcast of 1938
214: Boys Town
215: Carefree
216: Hotel du Nord
217: Jezebel
218: Port of Shadows
219: Test Pilot
220: You and Me
221:  The Stars Look Down
222:  Traktoristy
223:  Fantasia
224:  The Great Dictator
225:  Knute Rockne All American
226:  The Shop Around the Corner
227:  Sergeant York
228:  That Hamilton Woman
229:  Two-Faced Woman
230:  The Wolf Man
231:  Bambi
232:  The Magnificent Ambersons
233:  Prelude to War
234:  I Walked with a Zombie
235:  The More the Merrier
236:  Saludos Amigos
237:  Stormy Weather
238:  This Is the Army
239:  Marèa Candelaria (Xochimilco)
240:  National Velvet
241: Thirty Seconds Over Tokyo
242: The Clock
243: Stairway to Heaven
244: Notorious
245: Bowery Buckaroos
246: The Stranger
247: Gentleman's Agreement
248: Mine Own Executioner
249: Mourning Becomes Electra
250: Quai des Orfèvres
251: From Hand to Mouth
252: The Ace of Hearts
253: Peter Pan
254: Clash of the Wolves
255: Lady Windermere's Fan
256: Napoleon
257: Upstream
258: The Wedding March
259: Asphalt
260: Disraeli
261: Broken Lullaby
262: City Lights
263: Cimarron
264: Dishonored
265: The Champ
266: Le grand jeu
267: Movie Crazy
268: Shanghai Express
269: Smilin' Through
270: The Crowd Roars
271: Tarzan the Ape Man
272: State Fair
273: The Emperor Jones
274: The Power and the Glory
275: Two Seconds
276: The Band Concert
277: The Divorcee
278: The Lost Patrol
279: China Seas
280: Bad Girl
281: Flash Gordon
282: Night Must Fall
283: To New Shores
284: Alexander's Ragtime Band
285: The Childhood of Maxim Gorky
286: Drums
287: The Great Waltz
288: Sweeethearts
289: Three Comrades
290: Ninotchka
291: The Wizard of Oz
292: Young Mr. Lincoln
293: The Biscuit Eater
294: Strike Up the Band
295: Blossoms in the Dust
296: Folies Bergère de Paris
297: Naughty Marietta
298: A Night at the Opera
299: Scrooge
300: The Wedding Night
301: Desert Victory
302: Watch on the Rhine
303: The Seventh Victim
304: To Be or Not to Be
305: Dead of Night
306: A Corner in Wheat
307: The Pride of the Yankees
308: The Man Who Came to Dinner
309: Tunisian Victory
310: I See a Dark Stranger
311: The Best Years of Our Lives
312: Holy Matrimony
313: La porta del cielo
314: A Walk in the Sun
315: Anna and the King of Siam
316: My Darling Clementine
317: The Battle of Russia
318: The Best Years of Our Lives
319: Report from the Aleutians
320: A Song to Remember
321: David Golder
322: Four Sons
323: The Ball at the Anjo House
324: The Strong Man
325: Monsieur Vincent
326: One Week
327: The Son of the Sheik
328: Dèdèe d'Anvers
329: Boomerang!
330: Broadway Bill
331: Baby Face
332: Scarface
333: The Seventh Veil
334: She Done Him Wrong
335: Zoo in Budapest
336: Grand Hotel
337: Kiss of Death
338: Trouble in Paradise
339: Street Angel
340: Three Ages
341: Masquerade in Vienna
342: Alice Adams
343: Carnival in Flanders
344: The New Gulliver
345: Top Hat
346: Who Killed Cock Robin?
347: Anthony Adverse
348: Come and Get It
349: Mr. Deeds Goes to Town
350: Romeo and Juliet
351: Theodora Goes Wild
352: Camille
353: A Damsel in Distress
354: In Old Chicago
355: One Hundred Men and a Girl
356: The Prisoner of Zenda
357: Dance Program
358: The Citadel
359: The Lady Vanishes
360: The Rules of the Game
361: Down Argentine Way
362: The Southerner
363: One of Our Aircraft Is Missing
364: La perla
365: The Memphis Belle: A Story of a Flying Fortress
366: Saboteur
367: The Gang's All Here
368: Hangmen Also Die!
369: The Testimony
370: Going My Way
371: Phantom of the Opera
372: The Siege of the Alcazar
373: High Sierra
374: The Brothers and Sisters of the Toda Family
375: The Male Animal
376: Blood and Sand
377: Citizen Kane
378: Pinsocchio
379: The Siege of the Alcazar
380: The Brothers and Sisters of the Toda Family
381: La nao capitana
382: Salomè
383: The Damned
384: Mèdchen in Uniform
385: For Heaven's Sake
386: The Thief of Bagdad
387: Dreams That Money Can Buy
388: Mother Wore Tights
389: Possessed
390: Pastoral Symphony
391: Panic
392: It's a Wonderful Life
393: The Adventures of Robin Hood
394: The Cat Concerto
395: The Big Sleep
396: Body and Soul
397: King of Jazz
398: EARTH
399: Let There Be Light
400: For Lost Youth
401: Skippy
402: Forbidden
403: Dekigokoro
404: Sons of the Desert
405: Dames
406: Little Miss Marker
407: Man of Aran
408: Black Fury
409: Roberta
410: Follow the Fleet
411: Sisters of the Gion
412: Popeye the Sailor Meets Sindbad the Sailor
413: The Robber Symphony
414: The Story of a Cheat
415: The Trail of the Lonesome Pine
416: The Good Earth
417: They Won't Forget
418: Der zerbrochene Krug
419: The Adventures of Robin Hood
420: The Adventures of Tom Sawyer
421: Mother Carey's Chickens
422: A Man to Remember
423: Of Human Hearts
424: Too Much Johnson
425: Volga - Volga
426: Gunga Din
427: Dance, Girl, Dance
428: His Girl Friday
429: The Blood of Jesus
430: The Devil and Daniel Webster
431: The Well-Digger's Daughter
432: Cat People
433: People on the Alps
434: The Moon and Sixpence
435: Ossessione
436: The Pied Piper
437: Random Harvest
438: Kate & Leopold
439: Wake Island
440: Yankee Doodle Dandy
441: Hello Frisco, Hello
442: So Proudly We Hail!
443: Lifeboat
444: None But the Lonely Heart
445: And Then There Were None
446: The Three Caballeros
447: Wonder Man
448: The Chase
449: The Eagle with Two Heads
450: Miracle on 34th Street
451: Angelina
452: The Italian
453: High and Dizzy
454: Foolish Wives
455: He Who Gets Slapped
456: Wild Oranges
457: Grass: A Nation's Battle for Life
458: The Black Pirate
459: Laugh, Clown, Laugh
460: The Big Trail
461: Little Caesar
462: The Sin of Madelon Claudet
463: Dr. Jekyll and Mr. Hyde
464: The Song of Night
465: Footlight Parade
466: Flying Down to Rio
467: Going Hollywood
468: Wild Boys of the Road
469: The Gay Divorcee
470: Curly Top
471: The Lower Depths
472: Lloyd's of London
473: Modern Times
474: Captains Courageous
475: Marked Woman
476: Snow White and the Seven Dwarfs
477: Stage Door
478: Bringing Up Baby
479: Four Daughters
480: Vivacious Lady
481: Dark Victory
482:  The Rains Came
483:  The Spy in Black
484:  On His Own
485:  The Bank Dick
486:  The Letter
487:  Here Comes Mr. Jordan
488:  How Green Was My Valley
489:  The Little Foxes
490:  Love on the Dole
491:  Road to Zanzibar
492:  Tom Dick and Harry
493:  Der groèe Kènig
494:  The Hard Way
495:  My Gal Sal
496:  Victory Through Air Power
497:  The Curse of the Cat People
498:  Meet Me in St. Louis
499:  Murder, My Sweet
500:  Since You Went Away
501: House of Dracula
502: Make Mine Music
503: The Paleface
504: I Became a Criminal
505: L'amore
506: The Emperor Waltz
507: The Mating of Millie
508: Bedlam
509: The Captive Heart
510: The Diary of a Chambermaid
511: Enamorada
512: The Yearling
513: The Fugitive
514: Green Dolphin Street
515: The Sin of Harold Diddlebock
516: Rhapsody in Blue
517: The Body Snatcher
518: That Lady in Ermine
519: Apartment for Peggy
520: The Last Chance
521: Anni difficili
522: The Big Clock
523: Without Pity
524: The Street with No Name
525: Yoru no onnatachi
526: All the King's Men
527: An Act of Murder
528: Symphony of Life
529: All My Sons
530: Angels' Alley
531: Berlin Express
532: Easter Parade
533: I Remember Mama
534: To the Ends of the Earth
535: Command Decision
536: He Walked by Night
537: June Bride
538: Macbeth
539: The Mill on the Floss
540: Red River
541: Salèn Mèxico
542: La Terra Trema
543: You Were Meant for Me
544: Torment
545: Call Northside 777
546: The Walls of Malapaga
547: Every Girl Should Be Married
548: A Foreign Affair
549: Melody Time
550: Rope
551: Sitting Pretty
552: The Snake Pit
553: Sorry, Wrong Number
554: Yellow Sky
555: Adam's Rib
556: Act of Violence
557: Cry of the City
558: Road House
559: The Search
560: Station West
561: La mies es mucha
562: Louisiana Story
563: The Naked City
564: Bitter Rice
565: Begone Dull Care
566: The Blue Lagoon
567: In the Name of the Law
568: The Hidden Room
569: A Hen in the Wind
570: The Winslow Boy
571: The Pirate
572: Force of Evil
573: One Wonderful Sunday
574: The Winslow Boy
575: The Barkleys of Broadway
576: Portrait of Jennie
577: Mr. Blandings Builds His Dream House
578: Bud Abbott Lou Costello Meet Frankenstein
579: Fort Apache
580: Little Women
581: Mi adorado Juan
582: Passport to Pimlico
583: Sands of Iwo Jima
584: Broken Arrow
585: Brigada criminal
586: Born Yesterday
587: Cyrano de Bergerac
588: Strange Deception
589: It Happens Every Spring
590: Manon
591: Pinky
592: Prince of Foxes
593: The Window
594: Story of a Love Affair
595: D.O.A.
596: Battleground
597: I Was a Male War Bride
598: Come to the Stable
599: Chicago Deadline
600: In the Good Old Summertime
601: Scene of the Crime
602: The Quiet Duel
603: The Stratton Story
604: Whisky Galore
605: Cinderella
606: Devil's Doorway
607: Lost Boundaries
608: The Passionate Friends
609: House of Strangers
610: Home of the Brave
611: Madame Bovary
612: Neptune's Daughter
613: The Queen of Spades
614: The Set-Up
615: Take Me Out to the Ball Game
616: Edward and Caroline
617: Edge of Doom
618: The Heiress
619: Give Us This Day
620: Intruder in the Dust
621: Under Capricorn
622: Das doppelte Lottchen
623: On the Town
624: Gone to Earth
625: All About Eve
626: The Asphalt Jungle
627: Mighty Joe Young
628: Caged
629: Quartet
630: That Forsyte Woman
631: Jolson Sings Again
632: The Hasty Heart
633: The Fall of Berlin
634: Edward, My Son
635: The Black Rose
636: The Blue Lamp
637: Champion
638: Wuya yu maque
639: Annie Get Your Gun
640: Occupe-toi d'Amèlie..!
641: Gun Crazy
642: Highly Dangerous
643: Trio
644: Diary of a Country Priest
645: In a Lonely Place
646: The Lawless
647: Rocketship X-M
648: Angels in the Outfield
649: Kaunis Veera eli ballaadi Saimaalta
650: My Blue Heaven
651: The Munekata Sisters
652: Summer Stock
653: Victims of Sin
654: The African Queen
655: Cry, the Beloved Country
656: Montana
657: Susana
658: Bellissima
659: The Browning Version
660: David and Bathsheba
661: An American in Paris
662: Cops and Robbers
663: Awaara
664: The Blue Veil
665: Bright Victory
666: The Great Manhunt
667: Mister 880
668: Operation Disaster
669: Panic in the Streets
670: Rio Grande
671: The Sound of Fury
672: The Trio
673: No Way Out
674: The West Point Story
675: Carmen Comes Home
676: The Jackpot
677: Seven Days to Noon
678: Treasure Island
679: The Magnificent Yankee
680: The Men
681: Repast
682: Side Street Story
683: Alice in Wonderland
684: Cèrcel de mujeres
685: Father's Little Dividend
686: El hombre sin rostro
687: King Solomon's Mines
688: Trio
689: Stage Fright
690: Winchester '73
691: When Willie Comes Marching Home
692: The Day the Earth Stood Still
693: Death of a Salesman
694: Detective Story
695: Five
696: The Great Caruso
697: Kon-Tiki
698: Trio
699: Attenção! Bandits!
700: Four Ways Out
701: Europe '51
702: Fourteen Hours
703: Here Comes the Groom
704: The Man with a Cloak
705: Pandora and the Flying Dutchman
706: A Place in the Sun
707: On the Riviera
708: Strangers on a Train
709: The Steel Helmet
710: Children of Hiroshima
711: The Happy Time
712: Room for One More
713: The Snows of Kilimanjaro
714: Quo Vadis
715: Breaking the Sound Barrier
716: Encore
717: High Noon
718: Neighbours
719: The Witch
720: People Will Talk
721: Der Verlorene
722: 5 Fingers
723: Carrie
724: Come Back, Little Sheba
725: White Mane
726: Ivanhoe
727: The Man in the White Suit
728: The Marrying Kind
729: My Son John
730: Phone Call from a Stranger
731: Miracle in Milan
732: The Magic Box
733: Beauties of the Night
734: Siraa Fil-Wadi
735: The Little World of Don Camillo
736: The People Against O'Hara
737: The River
738: When Worlds Collide
739: The Lavender Hill Mob
740: Million Dollar Mermaid
741: My Cousin Rachel
742: The Naked Spur
743: Scaramouche
744: Singin' in the Rain
745: The Mating Season
746: Too Young to Kiss
747: Robinson Crusoe
748: The Big Sky
749: El bruto
750: Waiting Women
751: Limelight
752: The Member of the Wedding
753: Pat and Mike
754: A Phantasy
755: Saturday's Hero
756: Teresa
757: The Well
758: Above and Beyond
759: The Miracle of Our Lady of Fatima
760: Flavor of Green Tea Over Rice
761: El
762: The Captain's Paradise
763: The Thief
764: A Streetcar Named Desire
765: Feudin' Fools
766: The Golden Coach
767: Eaux d'artifice
768: The Overcoat
769: Punktchen and Anton
770: Sudden Fear
771: Hans Christian Andersen
772: Barabbas
773: The Moon Is Blue
774: Call Me Madam
775: The Story of Three Loves
776: Torch Song
777: The Red Badge of Courage
778: Two Cents Worth of Hope
779: Man on a Tightrope
780: Mexican Bus Ride
781: The Proud and the Beautiful
782: Il segno di Venere
783: Wife
784: Calamity Jane
785: The Stranger in Between
786: Viva Zapata!
787: I vinti
788: Little Boy Lost
789: Loose in London
790: The Man Between
791: Othello
792: With a Song in My Heart
793: The Beast from 20,000 Fathoms
794: Welcome Mr. Marshall!
795: Beneath the 12-Mile Reef
796: The Cruel Sea
797: Genevieve
798: It Came from Outer Space
799: Shane
800: Statues also Die
801: Indiscretion of an American Wife
802: How to Marry a Millionaire
803: Gate of Hell
804: The Little Kidnappers
805: Mogambo
806: Neapolitans in Milan
807: Bread, Love and Dreams
808: Peter Pan
809: The Sun Shines Bright
810: The Actress
811: Carne de horca
812: The Conquest of Everest
813: The Glass Wall
814: House of Wax
815: The Hitch-Hiker
816: Kiss Me Kate
817: Lili
818: The Living Desert
819: Carne de horca
820: The Adultress
821: Desert Rats
822: Knights of the Round Table
823: Pickup on South Street
824: The Fiancee
825: Sadko
826: The White Reindeer
827: The Band Wagon
828: Duck Amuck
829: Duck Dodgers in the 24èth Century
830: Invaders from Mars
831: The Lady Without Camelias
832: Ugetsu
833: The Egyptian
834: Broken Lance
835: Mr. Hulot's Holiday
836: The Caine Mutiny
837: Executive Suite
838: Salt of the Earth
839: There's No Business Like Show Business
840: Daddy Long Legs
841: Rouge et noir
842: Seven Brides for Seven Brothers
843: Blinkity Blank
844: The Garden of Women
845: Magnificent Obsession
846: The Count of Monte Cristo
847: Dial M for Murder
848: Liliomfi
849: The Long, Long Trailer
850: A Star Is Born
851: Them!
852: This Island Earth
853: Sound of the Mountain
854: The Desperate Hours
855: Carmen Jones
856: The Dam Busters
857: The Count of Monte Cristo
858: East of Eden
859: Siraa Fil-Wadi
860: Scarlet Week
861: Doctor in the House
862: The Glenn Miller Story
863: It Should Happen to You
864: Frisky
865: Susan Slept Here
866: The Vanishing Prairie
867: The War of the Worlds
868: 20,000 Leagues Under the Sea
869: Brigadoon
870: Fear
871: The River and Death
872: True Friends
873: Le amiche
874: Il Bidone
875: Young Bess
876: Late Chrysanthemums
877: The Big Knife
878: The End of the Affair
879: Ana-ta-han
880: The Barefoot Contessa
881: Knock on Wood
882: Blackboard Jungle
883: The Court-Martial of Billy Mitchell
884: Devdas
885: The Bridges at Toko-Ri
886: The Crucified Lovers
887: Chronicle of Poor Lovers
888: Hell and High Water
889: The Miracle of Marcelino
890: The Sheep Has Five Legs
891: The Gold of Naples
892: Bad Day at Black Rock
893: Creature from the Black Lagoon
894: Lovers, Happy Lovers!
895: Phffft
896: Doctor at Sea
897: Kiss Me Deadly
898: Lady and the Tramp
899: Strategic Air Command
900: The Violent Men
901: Interrupted Melody
902: Marty
903: The Night of the Hunter
904: Pete Kelly's Blues
905: The Tender Trap
906: Trial
907: Yèkihi
908: The Red Balloon
909: Carousel
910: Footsteps in the Fog
911: The Grand Maneuver
912: I'll Cry Tomorrow
913: Wild Love
914: Oklahoma!
915: One Froggy Evening
916: Smiles of a Summer Night
917: The Bespoke Overcoat
918: Raquel's Shoeshiner
919: The Criminal Life of Archibaldo de la Cruz
920: I Live in Fear: Record of a Living Being
921: She Was Like a Wild Chrysanthemum
922: Shree 420
923: Sissi
924: Floating Clouds
925: The Catered Affair
926: Guys and Dolls
927: The Grand Maneuver
928: Love Me or Leave Me
929: Around the World in Eighty Days
930: Bhowani Junction
931: Main Street
932: It's Always Fair Weather
933: The Kentuckian
934: La 'Moara cu noroc'
935: Picnic
936: The Prisoner
937: Rebel Without a Cause
938: Street of Shame
939: The Bad Seed
940: Blonde Sinner
941: Jedda the Uncivilized
942: A Kid for Two Farthings
943: The Ladykillers
944: The Long Gray Line
945: Stella
946: Taira Clan Saga
947: The Trouble with Harry
948: Autumn Leaves
949: Baby Doll
950: A Man Called Peter
951: The Man with the Golden Arm
952: Not as a Stranger
953: The Seven Year Itch
954: To Catch a Thief
955: The Unknown Soldier
956: Attack
957: Aparajito
958: The Brave One
959: Bus Stop
960: The Rocket from Calabuch
961: High Society
962:  Gervaise
963: The King and I
964: Crashing Las Vegas
965: Invention of the Dance
966: Seven Men from Now
967:  Somebody Up There Likes Me
968: The Man in the Gray Flannel Suit
969: The Forty-first
970: The Great Locomotive Chase
971:  Guendalina
972: Invasion of the Body Snatchers
973: The Bachelor
974: Sissi: The Young Empress
975: Last Pair Out
976: Tea and Sympathy
977: The Great Man
978: Giant
979: Meet Me in Las Vegas
980: An Eye for an Eye
981: Secrets of Life
982: The Searchers
983: Lust for Life
984: ...And God Created Woman
985: The Eddy Duchin Story
986: The Man Who Never Was
987: Flowing
988: The Captain from Kèpenick
989: Le mystère Picasso
990: Cien
991: Come Back, Africa
992: Donatella
993: The Court Jester
994: The Solid Gold Cadillac
995: The Rainmaker
996: The Spanish Gardener
997: The Silent World
998: The Man Who Knew Too Much
999: Embajadores en el infierno
1000: Forbidden Planet
```
