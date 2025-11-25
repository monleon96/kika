/**
 * File type detection utilities
 * Based on the Streamlit implementation's detection logic
 * 
 * Strategy: Try parsing content-based detection to identify ACE vs ENDF
 * Similar to Python backend but implemented in TypeScript for faster feedback
 */

import type { FileType } from '../types/file';

/**
 * Check if a ZAID is valid
 * ZAID format: ZZZAAA (Z = atomic number, A = mass number)
 */
export function isValidZaid(zaid: string | number | null | undefined): boolean {
  if (!zaid) return false;

  try {
    const zaidInt = typeof zaid === 'string' ? parseInt(zaid, 10) : zaid;

    // ZAID should be positive and reasonable (1000 to 999999)
    if (zaidInt <= 0 || zaidInt > 999999) {
      return false;
    }

    // Extract Z (atomic number) and A (mass number)
    const z = Math.floor(zaidInt / 1000);
    const a = zaidInt % 1000;

    // Basic validation: Z should be 1-118 (known elements)
    if (z < 1 || z > 118) {
      return false;
    }

    // A should be reasonable (0-400, including metastable states)
    if (a > 400) {
      return false;
    }

    return true;
  } catch {
    return false;
  }
}

/**
 * Extract ZAID from ACE file header
 * ACE format first line typically: ZAID AWR TEMP COMMENT
 * Returns null if parsing fails
 */
function extractAceZaid(content: string): number | null {
  try {
    const firstLine = content.split('\n')[0]?.trim();
    if (!firstLine) return null;

    // ACE first line has format: ZAID AWR TEMP DATE/COMMENT
    // Example: "  1001.02c   1.00783 -1.12606  0"
    const parts = firstLine.split(/\s+/);
    if (parts.length < 3) return null;

    // First field is typically ZAID with possible 'c' suffix
    const zaidStr = parts[0].replace(/[cCeE]$/, ''); // Remove 'c' or 'e' suffix
    const zaid = parseFloat(zaidStr);
    
    return isFinite(zaid) ? Math.round(zaid) : null;
  } catch {
    return null;
  }
}

/**
 * Extract ZAID from ENDF file header
 * ENDF uses MAT number which often encodes ZAID
 * Returns null if parsing fails
 */
function extractEndfZaid(content: string): number | null {
  try {
    const lines = content.split('\n');
    
    // ENDF lines are typically 75+ characters with fixed format
    // MAT is in columns 67-70 (Fortran columns 67-70, 0-indexed: 66-69)
    // MF in 71-72, MT in 73-75, line number in 76-80
    
    // Look for characteristic ENDF patterns in first few lines
    for (const line of lines.slice(0, 20)) {
      // ENDF lines should be at least 66 chars (data section)
      if (line.length < 66) continue;
      
      // Extract MAT number (columns 67-70, 0-indexed: 66-69)
      const matStr = line.substring(66, 70).trim();
      const mat = parseInt(matStr, 10);
      
      // MAT should be a valid positive integer
      if (!isFinite(mat) || mat <= 0) continue;
      
      // For most ENDF libraries:
      // MAT = Z*100 + A (approximately)
      // Or MAT = ZZZAAA format where ZZZ is atomic number, AAA is mass number
      // Common range: 100-99999
      if (mat < 100 || mat > 99999) continue;
      
      // Try to extract ZAID from MAT
      // Common formats:
      // 1. MAT = Z * 25 + offset (ENDF/B-VII and earlier)
      // 2. MAT = ZZAAA (more recent, where ZZ=atomic number, AAA=mass)
      // 3. MAT = ZZZ0AA (padded format)
      
      // Try format: ZZAAA (most common modern format)
      // Extract potential Z (atomic number) and A (mass)
      let zaid: number;
      
      if (mat >= 1000) {
        // Format likely ZZAAA or ZZZAA
        const z = Math.floor(mat / 1000); // First 1-2 digits
        const a = mat % 1000; // Last 3 digits
        
        if (z >= 1 && z <= 118 && a >= 0 && a <= 400) {
          zaid = z * 1000 + a;
          if (isValidZaid(zaid)) {
            return zaid;
          }
        }
      } else {
        // Format might be ZAAA for light elements
        const z = Math.floor(mat / 100);
        const a = mat % 100;
        
        if (z >= 1 && z <= 118 && a >= 0 && a <= 400) {
          zaid = z * 1000 + a;
          if (isValidZaid(zaid)) {
            return zaid;
          }
        }
      }
    }
    
    return null;
  } catch {
    return null;
  }
}

/**
 * Detect if file content looks like ACE format
 * ACE files typically have:
 * - First line: ZAID AWR TEMP DATE
 * - Multiple lines with scientific notation numbers
 * - Specific structure with table markers
 */
function looksLikeACE(content: string): boolean {
  if (!content || content.length < 50) return false;

  const lines = content.split('\n').slice(0, 50); // Check first 50 lines
  
  // Try to extract ZAID
  const zaid = extractAceZaid(content);
  if (!zaid || !isValidZaid(zaid)) {
    return false;
  }

  // Check for characteristic ACE patterns
  // ACE files have lots of scientific notation
  let scientificNotationCount = 0;
  let nonCommentLines = 0;

  for (const line of lines.slice(0, 30)) {
    if (line.trim().startsWith('/') || line.trim().startsWith('-')) {
      // Likely a comment or separator
      continue;
    }
    
    nonCommentLines++;
    
    // Count scientific notation patterns
    const matches = (line.match(/[-+]?\d+\.\d+[Ee][-+]?\d+/g) || []).length;
    scientificNotationCount += matches;
  }

  // ACE files should have high density of scientific notation
  // At least 5+ occurrences in first 30 lines
  return scientificNotationCount >= 5;
}

/**
 * Detect if file content looks like ENDF-6 format
 * ENDF files have:
 * - Fixed width columns (typically 80 characters)
 * - Line ending pattern: MAT(4) MF(2) MT(3) LINE(5)
 * - Specific columnar structure with numeric data
 */
function looksLikeENDF(content: string): boolean {
  if (!content || content.length < 100) return false;

  const lines = content.split('\n').slice(0, 100); // Check first 100 lines
  
  // Check for ENDF line format patterns
  let endfLineMatches = 0;
  let totalLinesChecked = 0;
  let hasValidMat = false;

  for (const line of lines) {
    // Skip empty lines
    if (line.trim().length === 0) continue;
    
    // ENDF lines are typically 66-81 characters
    // Data section: 66 chars, MAT/MF/MT/LINE: 14 chars
    if (line.length < 66) continue;
    
    totalLinesChecked++;

    // Check for ENDF line ending pattern
    // Format: ...data (66 chars)... MAT(4) MF(2) MT(3) LINE(5)
    // Columns (0-indexed): data(0-65), MAT(66-69), MF(70-71), MT(72-74), LINE(75-79)
    
    // Extract the trailing section (last 14 characters)
    const trailer = line.substring(Math.max(0, line.length - 14)).trim();
    
    // ENDF trailer pattern: 4-digit MAT, 1-2 digit MF, 1-3 digit MT, up to 5-digit line number
    // More relaxed pattern to catch variations
    const endfPattern = /^\d{3,4}\s+\d{1,2}\s+\d{1,3}(\s+\d{1,5})?$/;
    if (endfPattern.test(trailer)) {
      endfLineMatches++;
      
      // Also check if MAT is valid
      if (!hasValidMat && line.length >= 70) {
        const matStr = line.substring(66, 70).trim();
        const mat = parseInt(matStr, 10);
        if (isFinite(mat) && mat > 0) {
          hasValidMat = true;
        }
      }
    }
    
    // Alternative: Check for ENDF-6 characteristic patterns in data section
    // ENDF often has scientific notation with specific formats
    if (line.includes('E+') || line.includes('E-') || line.includes('e+') || line.includes('e-')) {
      // Lines with exponential notation are common in ENDF
      const expCount = (line.match(/[Ee][+-]\d/g) || []).length;
      if (expCount >= 2) {
        endfLineMatches += 0.5; // Partial match
      }
    }
  }

  // Require:
  // 1. At least 5 lines that match ENDF patterns
  // 2. Pattern matches make up at least 25% of checked lines
  // 3. Found at least one valid MAT number
  const threshold = Math.max(5, totalLinesChecked * 0.25);
  return totalLinesChecked > 0 && endfLineMatches >= threshold && (hasValidMat || endfLineMatches >= 10);
}

/**
 * Detect file type by analyzing file content
 * Strategy similar to Python Streamlit version:
 * 1. Try to detect ENDF-6 format
 * 2. Try to detect ACE format
 * 3. Return the most likely type or null if ambiguous
 * 
 * Note: Backend will do full validation. This is just for user feedback.
 */
export function detectFileType(filename: string, content: string): FileType | null {
  // Content-based detection (most reliable)
  const contentIsENDF = looksLikeENDF(content);
  const contentIsACE = looksLikeACE(content);

  // If only one matches, use that
  if (contentIsENDF && !contentIsACE) {
    return 'endf';
  }

  if (contentIsACE && !contentIsENDF) {
    return 'ace';
  }

  // If both or neither match, try extension as fallback
  const ext = filename.toLowerCase().split('.').pop() || '';
  
  // Common ACE extensions
  const aceExtensions = ['ace', '02c', '03c', '20c', '21c', '70c', '80c', '12c', '32c', '04c'];
  
  // Common ENDF extensions
  const endfExtensions = ['endf', 'endf6', 'jeff', 'jendl', 'tendl', 'txt'];
  
  if (aceExtensions.includes(ext)) {
    return 'ace';
  }
  
  if (endfExtensions.includes(ext)) {
    return 'endf';
  }
  
  // Ambiguous or unknown - return null
  // Backend will attempt auto-detection by actually parsing
  return null;
}

/**
 * Format file size in human-readable format
 */
export function formatFileSize(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = bytes;
  let unitIndex = 0;
  
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  
  return `${size.toFixed(1)} ${units[unitIndex]}`;
}

/**
 * Generate a unique file ID
 */
export function generateFileId(): string {
  return `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
}
