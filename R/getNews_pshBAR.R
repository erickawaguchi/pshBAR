#' @export
getNews_pshBAR <- function(...) {
  newsfile <- file.path(system.file(package = "pshBAR"), "NEWS")
  file.show(newsfile)
}
