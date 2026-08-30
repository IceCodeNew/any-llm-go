package openai

import (
	"fmt"

	"github.com/mozilla-ai/any-llm-go/errors"
)

func invalid(message string) error {
	return errors.NewInvalidRequestError(providerName, fmt.Errorf("%s", message))
}

func reportStreamError(errs chan<- error, err error) {
	select {
	case errs <- err:
	default:
	}
}
