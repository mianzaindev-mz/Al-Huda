/**
 * Islamic AI Assistant - Robust Markdown Formatter v4.0
 * 
 * Complete rewrite to avoid placeholder issues.
 * Processes all formatting in a single pass with proper escaping.
 */

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function escapeHtml(text) {
    if (!text) return '';
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}

// ============================================================================
// QURAN VERSE RENDERER
// ============================================================================

function renderQuranBox(reference, arabic, translation) {
    const hasArabic = arabic && arabic.length > 0;
    const hasTranslation = translation && translation.length > 0;

    return `
<div class="quran-verse">
    <div class="quran-verse-header">
        <span class="quran-icon">📖</span>
        <span class="quran-reference">Quran ${escapeHtml(reference)}</span>
    </div>
    ${hasArabic ? `<div class="quran-text">${escapeHtml(arabic)}</div>` : ''}
    ${hasTranslation ? `<div class="quran-translation">${formatInlineText(translation)}</div>` : ''}
</div>`;
}

// ============================================================================
// HADITH RENDERER
// ============================================================================

function renderHadithBox(reference, hadithText, narrator) {
    const hasText = hadithText && hadithText.length > 0;
    const hasNarrator = narrator && narrator.length > 0;
    const cleanText = hasText ? hadithText.replace(/^["'"]|["'"]$/g, '') : '';

    return `
<div class="hadith-box">
    <div class="hadith-header">
        <span class="hadith-icon">📜</span>
        <span class="hadith-reference">${escapeHtml(reference)}</span>
    </div>
    ${hasText ? `<div class="hadith-text">"${formatInlineText(cleanText)}"</div>` : ''}
    ${hasNarrator ? `<div class="hadith-narrator">${formatInlineText(narrator)}</div>` : ''}
</div>`;
}

// ============================================================================
// TABLE RENDERER
// ============================================================================

function renderTable(headerRow, bodyRows) {
    const headers = headerRow.split('|').map(h => h.trim()).filter(h => h);
    const rows = bodyRows.trim().split('\n').map(row =>
        row.split('|').map(cell => cell.trim()).filter(cell => cell)
    );

    let html = '<div class="table-wrapper"><table>\n<thead>\n<tr>\n';
    headers.forEach(header => {
        html += `<th>${formatInlineText(header)}</th>\n`;
    });
    html += '</tr>\n</thead>\n<tbody>\n';

    rows.forEach(row => {
        html += '<tr>\n';
        for (let i = 0; i < headers.length; i++) {
            const cellContent = row[i] || '—';
            html += `<td>${formatInlineText(cellContent)}</td>\n`;
        }
        html += '</tr>\n';
    });

    html += '</tbody>\n</table>\n</div>';
    return html;
}

// ============================================================================
// INLINE TEXT FORMATTER (bold, italic, code, links)
// ============================================================================

function formatInlineText(text) {
    if (!text) return '';
    let result = text;

    // Bold + Italic (***text***) - allow anything except newline
    result = result.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    // Bold (**text**) - allow anything except newline, non-greedy
    result = result.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Italic (*text*) - avoid list markers at start of line, single asterisks
    result = result.replace(/(?<!\*)\*([^*\n]+?)\*(?!\*)/g, '<em>$1</em>');
    // Bold (__text__) - allow anything except newline
    result = result.replace(/__(.+?)__/g, '<strong>$1</strong>');
    // Italic (_text_) - single underscores
    result = result.replace(/(?<!_)_([^_\n]+?)_(?!_)/g, '<em>$1</em>');
    // Inline code (`code`)
    result = result.replace(/`([^`]+?)`/g, '<code>$1</code>');
    // Links [text](url)
    result = result.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

    return result;
}

// ============================================================================
// MAIN FORMATTER FUNCTION
// ============================================================================

function formatMarkdownEnhanced(text) {
    if (!text) return '';

    let result = text;

    // ========================================================================
    // STEP 0: Clean up any stray placeholder patterns or unwanted text
    // ========================================================================

    // Remove any remaining placeholder patterns from any source
    result = result.replace(/___[A-Z]+_?PLACEHOLDER_?\d+___/gi, '');
    result = result.replace(/___[A-Z]+PLACEHOLDER\d+___/gi, '');

    // Remove stray text patterns
    result = result.replace(/\bdark_mode\b/g, '');
    result = result.replace(/\[Box\]/gi, '');
    result = result.replace(/\[Small Green Box\]/gi, '');
    result = result.replace(/\[Blue\/Purple Box\]/gi, '');
    result = result.replace(/\[Citation needed\]/gi, '');
    result = result.replace(/\[TODO\]/gi, '');
    result = result.replace(/\{\{[^}]+\}\}/g, '');
    result = result.replace(/\$\{[^}]+\}/g, '');

    // ========================================================================
    // STEP 1: Process Quran verses - [Quran:Ref|Arabic|Translation]
    // Uses bracket-counting parser to handle nested brackets in translations
    // ========================================================================

    function parseQuranVerses(text) {
        let result = text;
        const quranPattern = /\[Quran:/g;
        let match;
        const replacements = [];

        while ((match = quranPattern.exec(text)) !== null) {
            const startIdx = match.index;
            let idx = startIdx + 7; // After "[Quran:"

            // Extract reference (until first |)
            let ref = '';
            while (idx < text.length && text[idx] !== '|') {
                ref += text[idx];
                idx++;
            }
            idx++; // Skip |

            // Extract Arabic (until second |)
            let arabic = '';
            while (idx < text.length && text[idx] !== '|') {
                arabic += text[idx];
                idx++;
            }
            idx++; // Skip |

            // Extract translation - count brackets to find the TRUE closing ]
            let translation = '';
            let bracketDepth = 1; // We're inside the outer [Quran:...]

            while (idx < text.length && bracketDepth > 0) {
                const char = text[idx];
                if (char === '[') {
                    bracketDepth++;
                    translation += char;
                } else if (char === ']') {
                    bracketDepth--;
                    if (bracketDepth > 0) {
                        translation += char; // Inner bracket, include it
                    }
                    // If bracketDepth === 0, this is the closing bracket, don't include
                } else {
                    translation += char;
                }
                idx++;
            }

            // Store replacement
            const fullMatch = text.substring(startIdx, idx);
            const replacement = renderQuranBox(ref.trim(), arabic.trim(), translation.trim());
            replacements.push({ original: fullMatch, replacement: replacement });
        }

        // Apply replacements in reverse order to preserve indices
        for (let i = replacements.length - 1; i >= 0; i--) {
            result = result.replace(replacements[i].original, replacements[i].replacement);
        }

        return result;
    }

    result = parseQuranVerses(result);

    // Simple Quranic reference (Qur'an X:Y)
    result = result.replace(/\(Qur'an\s+([^)]+)\)/gi, (match, ref) => {
        return `<span class="quran-inline-ref">(Quran ${escapeHtml(ref.trim())})</span>`;
    });

    // ========================================================================
    // STEP 2: Process Hadith - [Hadith:Ref|"Text"|Narrator]
    // ========================================================================

    result = result.replace(/\[Hadith:([^|\]]+)\|([^|]*)\|([^\]]*)\]/g, (match, ref, hadithText, narrator) => {
        return renderHadithBox(ref.trim(), hadithText.trim(), narrator.trim());
    });

    // Simple hadith reference [Sahih Bukhari 1234]
    result = result.replace(/\[(Sahih (?:Bukhari|Muslim)|Sunan (?:Abu Dawud|Tirmidhi|Ibn Majah|al-Nasa'i)|Muwatta Malik)([^\]]*)\]/gi, (match, collection, number) => {
        return `<span class="hadith-inline-ref">[${escapeHtml(collection + number)}]</span>`;
    });

    // ========================================================================
    // STEP 3: Process tables - More flexible pattern matching
    // ========================================================================

    // Pattern 1: Standard markdown tables with pipes on both ends
    result = result.replace(/\n?\|(.+)\|\r?\n\|([-:\s|]+)\|\r?\n((?:\|.+\|\r?\n?)+)/g, (match, headerRow, separator, bodyRows) => {
        return '\n' + renderTable(headerRow, bodyRows) + '\n';
    });

    // Pattern 2: Tables without trailing pipes (some markdown variants)
    result = result.replace(/\n?\|(.+)\r?\n\|([-:\s|]+)\r?\n((?:\|.+\r?\n?)+)/g, (match, headerRow, separator, bodyRows) => {
        // Check if this looks like a table we haven't processed yet
        if (match.includes('<table>')) return match;
        return '\n' + renderTable(headerRow, bodyRows) + '\n';
    });

    // Pattern 3: Simple table format without outer pipes
    result = result.replace(/\n?([^\n|]+(?:\|[^\n|]+)+)\r?\n([-:\s]+(?:\|[-:\s]+)+)\r?\n((?:[^\n]+(?:\|[^\n]+)+\r?\n?)+)/g, (match, headerRow, separator, bodyRows) => {
        // Skip if already processed or doesn't look like a table
        if (match.includes('<table>') || !separator.match(/^[\s|:-]+$/)) return match;
        return '\n' + renderTable(headerRow, bodyRows) + '\n';
    });

    // ========================================================================
    // STEP 4: Process code blocks (```code```)
    // ========================================================================

    result = result.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre class="code-block"><code class="language-${lang || 'text'}">${escapeHtml(code.trim())}</code></pre>`;
    });

    // ========================================================================
    // STEP 5: Process blockquotes
    // ========================================================================

    result = result.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');
    result = result.replace(/<\/blockquote>\n<blockquote>/g, '\n');

    // ========================================================================
    // STEP 6: Process headings
    // ========================================================================

    result = result.replace(/^#### (.+?)$/gm, '\n<h4>$1</h4>\n');
    result = result.replace(/^### (.+?)$/gm, '\n<h3>$1</h3>\n');
    result = result.replace(/^## (.+?)$/gm, '\n<h2>$1</h2>\n');
    result = result.replace(/^# (.+?)$/gm, '\n<h1>$1</h1>\n');

    // ========================================================================
    // STEP 7: Process horizontal rules (--- or ***)
    // ========================================================================

    result = result.replace(/^---$/gm, '<hr>');
    result = result.replace(/^\*\*\*$/gm, '<hr>');

    // ========================================================================
    // STEP 8: Process lists
    // ========================================================================

    const lines = result.split('\n');
    const processedLines = [];
    let inList = false;
    let listType = null;
    let listItems = [];

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const trimmedLine = line.trim();

        // Check for unordered list item (- or *)
        const ulMatch = trimmedLine.match(/^[-*]\s+(.+)$/);
        // Check for ordered list item (1. 2. etc.)
        const olMatch = trimmedLine.match(/^\d+\.\s+(.+)$/);

        if (ulMatch || olMatch) {
            const content = ulMatch ? ulMatch[1] : olMatch[1];
            const currentType = ulMatch ? 'ul' : 'ol';

            if (!inList) {
                inList = true;
                listType = currentType;
                listItems = [content];
            } else if (listType !== currentType) {
                // Close previous list
                processedLines.push(`<${listType}>`);
                listItems.forEach(item => processedLines.push(`  <li>${formatInlineText(item)}</li>`));
                processedLines.push(`</${listType}>`);
                // Start new list
                listType = currentType;
                listItems = [content];
            } else {
                listItems.push(content);
            }
        } else {
            if (inList) {
                // Close current list
                processedLines.push(`<${listType}>`);
                listItems.forEach(item => processedLines.push(`  <li>${formatInlineText(item)}</li>`));
                processedLines.push(`</${listType}>`);
                inList = false;
                listType = null;
                listItems = [];
            }
            processedLines.push(line);
        }
    }

    // Close any remaining list
    if (inList) {
        processedLines.push(`<${listType}>`);
        listItems.forEach(item => processedLines.push(`  <li>${formatInlineText(item)}</li>`));
        processedLines.push(`</${listType}>`);
    }

    result = processedLines.join('\n');

    // ========================================================================
    // STEP 8.5: Apply inline formatting GLOBALLY (before paragraph wrapping)
    // ========================================================================

    // Apply inline formatting to text that's not inside HTML tags
    // Process line by line to handle text between/outside block elements
    const globalLines = result.split('\n');
    result = globalLines.map(line => {
        const trimmed = line.trim();
        // Skip lines that are fully HTML tags
        if (trimmed.startsWith('<') && trimmed.endsWith('>')) {
            // But still format content INSIDE heading tags
            line = line.replace(/<(h[1-4])>(.+?)<\/\1>/g, (match, tag, content) => {
                return `<${tag}>${formatInlineText(content)}</${tag}>`;
            });
            return line;
        }
        // For non-HTML lines, apply inline formatting directly
        if (!trimmed.startsWith('<')) {
            return formatInlineText(line);
        }
        return line;
    }).join('\n');

    // ========================================================================
    // STEP 9: Wrap paragraphs
    // ========================================================================

    const blocks = result.split(/\n\n+/);
    result = blocks.map(block => {
        block = block.trim();
        if (!block) return '';

        // Don't wrap if it's already a block element
        if (block.match(/^<(h[1-4]|ul|ol|blockquote|hr|div|table|pre)/)) {
            return block;
        }

        // Apply inline formatting and wrap in paragraph
        const formatted = formatInlineText(block.replace(/\n/g, ' '));
        return `<p>${formatted}</p>`;
    }).filter(b => b).join('\n');

    // ========================================================================
    // STEP 10: Final cleanup
    // ========================================================================

    // Remove excessive newlines
    result = result.replace(/\n{3,}/g, '\n\n');
    // Remove empty paragraphs
    result = result.replace(/<p>\s*<\/p>/g, '');
    // Clean up spacing
    result = result.replace(/>\s+</g, '><');
    // Ensure proper spacing around block elements
    result = result.replace(/(<\/(div|ul|ol|blockquote|table|pre)>)(?!\n)/g, '$1\n');
    result = result.replace(/(?<!\n)(<(div|ul|ol|blockquote|table|h[1-4]|pre))/g, '\n$1');

    return result.trim();
}

// ============================================================================
// EXPORT
// ============================================================================

window.formatMarkdownEnhanced = formatMarkdownEnhanced;
