package anyllm_test

import (
	"testing"

	anyllm "github.com/mozilla-ai/any-llm-go"
)

func TestRoleDeveloper(t *testing.T) {
	t.Parallel()

	if anyllm.RoleDeveloper != "developer" {
		t.Fatalf("RoleDeveloper = %q, want developer", anyllm.RoleDeveloper)
	}
}
