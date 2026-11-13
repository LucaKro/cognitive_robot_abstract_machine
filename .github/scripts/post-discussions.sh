#!/usr/bin/env bash
# Posts each discussion waiting in the posts directory to the repository's Discussions,
# in the order it was queued, and removes it from the notes branch once posted. A post
# whose kind is "comment" is added as a comment to the oldest discussion of its title in
# its category, which is started with the post's introduction when there is none. A post
# naming a category the repository lacks, or one GitHub does not take, is kept, and the
# run fails. Run by the posting workflow in a checkout of the notes branch, with gh
# signed in.
#
# Usage: post-discussions.sh <posts directory> <categories query> <create query>
#   <find query> <comment query> <owner/name> <notes branch> <commit script>
set -euo pipefail

POSTS="$1"
CATEGORIES_QUERY="$2"
CREATE_QUERY="$3"
FIND_QUERY="$4"
COMMENT_QUERY="$5"
REPOSITORY="$6"
NOTES_BRANCH="$7"
COMMIT_SCRIPT="$8"
OWNER="${REPOSITORY%%/*}"
NAME="${REPOSITORY#*/}"

# Creates a discussion and prints the field of it the jq filter names.
# Usage: create_discussion <category id> <title> <body> <jq filter>
create_discussion() {
  gh api graphql -F query=@"${CREATE_QUERY}" \
    -f repositoryId="${repository_id}" \
    -f categoryId="$1" \
    -f title="$2" \
    -f body="$3" \
    --jq "$4"
}

# Posts a queued discussion and prints its address.
# Usage: post_discussion <post> <category id>
post_discussion() {
  create_discussion "$2" "$(jq -r '.title' "$1")" "$(jq -r '.body' "$1")" \
    '.data.createDiscussion.discussion.url'
}

# Adds a queued comment to the discussion it names, started first when missing, and
# prints the comment's address.
# Usage: post_comment <post> <category id>
post_comment() {
  local title found discussion_id
  title="$(jq -r '.title' "$1")"
  found="$(gh api graphql -F query=@"${FIND_QUERY}" -f owner="${OWNER}" -f name="${NAME}" -f categoryId="$2")" || return 1
  discussion_id="$(jq -r --arg title "${title}" 'first(.data.repository.discussions.nodes[] | select(.title == $title) | .id) // empty' <<<"${found}")"
  if [ -z "${discussion_id}" ]; then
    discussion_id="$(create_discussion "$2" "${title}" "$(jq -r '.introduction' "$1")" '.data.createDiscussion.discussion.id')" || return 1
  fi
  gh api graphql -F query=@"${COMMENT_QUERY}" \
    -f discussionId="${discussion_id}" \
    -f body="$(jq -r '.body' "$1")" \
    --jq '.data.addDiscussionComment.comment.url'
}

shopt -s nullglob
waiting=("${POSTS}"/*.json)
if [ "${#waiting[@]}" -eq 0 ]; then
  exit 0
fi
found="$(gh api graphql -F query=@"${CATEGORIES_QUERY}" -f owner="${OWNER}" -f name="${NAME}")"
repository_id="$(jq -r '.data.repository.id' <<<"${found}")"
kept=0
for post in "${waiting[@]}"; do
  category="$(jq -r '.category' "${post}")"
  category_id="$(jq -r --arg name "${category}" '.data.repository.discussionCategories.nodes[] | select(.name == $name) | .id' <<<"${found}")"
  if [ -z "${category_id}" ]; then
    echo "kept ${post}: ${REPOSITORY} has no discussion category named ${category}" >&2
    kept=1
    continue
  fi
  if [ "$(jq -r '.kind // "discussion"' "${post}")" = "comment" ]; then
    posted=post_comment
  else
    posted=post_discussion
  fi
  if ! "${posted}" "${post}" "${category_id}"; then
    echo "kept ${post}: GitHub did not take it" >&2
    kept=1
    continue
  fi
  rm "${post}"
done
bash "${COMMIT_SCRIPT}" "Remove the discussions posted" "${NOTES_BRANCH}" "${POSTS}"
exit "${kept}"
