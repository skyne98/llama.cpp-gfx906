#!/bin/bash
# Script to migrate from local ggml to ggml-gfx906 fork as submodule

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  GGML to GGML-GFX906 Submodule Migration  ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Configuration
GGML_FORK_URL="https://github.com/skyne98/ggml-gfx906"
GGML_UPSTREAM_URL="https://github.com/ggerganov/ggml"
BRANCH_NAME="gfx906-optimizations"

# Step 1: Check current state
echo -e "${GREEN}Step 1: Checking current repository state...${NC}"
if [ ! -d ".git" ]; then
    echo -e "${RED}Error: Not in a git repository${NC}"
    exit 1
fi

if [ -d "ggml/.git" ]; then
    echo -e "${YELLOW}Warning: ggml is already a git submodule${NC}"
    echo "Current submodule URL:"
    git config --file .gitmodules submodule.ggml.url || echo "No submodule configuration found"
    read -p "Do you want to continue and update the submodule? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
fi

# Step 2: Backup current ggml if it exists
if [ -d "ggml" ] && [ ! -d "ggml/.git" ]; then
    echo -e "${GREEN}Step 2: Backing up current ggml directory...${NC}"
    cp -r ggml ggml.backup.$(date +%Y%m%d_%H%M%S)
    echo "Backup created: ggml.backup.$(date +%Y%m%d_%H%M%S)"
else
    echo -e "${YELLOW}Step 2: Skipping backup (ggml is already a submodule or doesn't exist)${NC}"
fi

# Step 3: Remove existing ggml
echo -e "${GREEN}Step 3: Removing existing ggml directory...${NC}"
if [ -d "ggml/.git" ]; then
    # It's a submodule
    git submodule deinit -f ggml
    git rm -f ggml
    rm -rf .git/modules/ggml
else
    # It's a regular directory
    git rm -rf ggml 2>/dev/null || rm -rf ggml
fi

# Step 4: Add ggml-gfx906 as submodule
echo -e "${GREEN}Step 4: Adding ggml-gfx906 as submodule...${NC}"
git submodule add ${GGML_FORK_URL} ggml
git submodule update --init --recursive

# Step 5: Set up the fork for development
echo -e "${GREEN}Step 5: Setting up ggml-gfx906 fork for development...${NC}"
cd ggml

# Add upstream remote
git remote add upstream ${GGML_UPSTREAM_URL} 2>/dev/null || echo "Upstream remote already exists"
git fetch upstream

# Create/checkout optimization branch
git checkout -b ${BRANCH_NAME} 2>/dev/null || git checkout ${BRANCH_NAME}

# Step 6: Copy GFX906-specific optimizations if they exist in backup
cd ..
if [ -d "ggml.backup."* ]; then
    BACKUP_DIR=$(ls -d ggml.backup.* | head -1)
    echo -e "${GREEN}Step 6: Checking for GFX906-specific files to preserve...${NC}"
    
    # Look for GFX906-specific files
    if find ${BACKUP_DIR} -name "*gfx906*" -o -name "*GFX906*" 2>/dev/null | grep -q .; then
        echo "Found GFX906-specific files:"
        find ${BACKUP_DIR} -name "*gfx906*" -o -name "*GFX906*" 2>/dev/null
        
        read -p "Copy these files to the new submodule? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            find ${BACKUP_DIR} -name "*gfx906*" -o -name "*GFX906*" -exec cp {} ggml/src/ggml-cuda/ \; 2>/dev/null || true
            echo "Files copied to ggml/src/ggml-cuda/"
        fi
    fi
else
    echo -e "${YELLOW}Step 6: No backup found to check for GFX906 files${NC}"
fi

# Step 7: Update CMakeLists.txt if needed
echo -e "${GREEN}Step 7: Checking CMakeLists.txt...${NC}"
if ! grep -q "add_subdirectory(ggml)" CMakeLists.txt; then
    echo -e "${YELLOW}Note: You may need to update CMakeLists.txt to properly include the ggml submodule${NC}"
    echo "Typical change needed:"
    echo "  add_subdirectory(ggml)"
fi

# Step 8: Create .gitmodules if it doesn't exist
if [ ! -f .gitmodules ]; then
    echo -e "${GREEN}Step 8: Creating .gitmodules...${NC}"
    cat > .gitmodules << EOF
[submodule "ggml"]
	path = ggml
	url = ${GGML_FORK_URL}
	branch = ${BRANCH_NAME}
EOF
fi

# Step 9: Commit changes
echo -e "${GREEN}Step 9: Preparing to commit changes...${NC}"
git add .gitmodules
git add ggml
echo ""
echo -e "${GREEN}Changes staged. Suggested commit command:${NC}"
echo "git commit -m \"feat: Migrate to ggml-gfx906 fork as submodule"
echo ""
echo "- Replace local ggml with submodule from ${GGML_FORK_URL}"
echo "- Set up for GFX906-specific optimizations"
echo "- Branch: ${BRANCH_NAME}\""

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}✅ Migration prepared successfully!${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "Next steps:"
echo "1. Review the staged changes: git status"
echo "2. Commit the changes: git commit -m '...'"
echo "3. Push to remote: git push"
echo ""
echo "To work on GGML optimizations:"
echo "  cd ggml"
echo "  git checkout ${BRANCH_NAME}"
echo "  # Make your changes"
echo "  git commit -am 'Your changes'"
echo "  git push origin ${BRANCH_NAME}"
echo ""
echo "To update the submodule in llama.cpp later:"
echo "  cd ggml && git pull origin ${BRANCH_NAME}"
echo "  cd .. && git add ggml"
echo "  git commit -m 'chore: Update ggml submodule'"