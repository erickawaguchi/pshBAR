#' @export
getNews_pshBAR <- function(...) {
  newsfile <- file.path(system.file(package = "crrBAR"), "NEWS")
  file.show(newsfile)
}
